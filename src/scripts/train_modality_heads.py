"""Train independent modality heads while keeping all expert encoders frozen."""

import argparse
import copy

import torch
import torch.nn as nn

from src.data.splits import load_binary_splits
from src.models.modality_heads import MODALITIES, ModalityPredictionHeads
from src.models.multimodal_model import EAVMultimodalFusionNetwork
from src.scripts.experiment_utils import (
    load_state_dict,
    make_loader,
    seed_everything,
    select_device,
)


def evaluate_heads(model, heads, loader, device):
    model.eval()
    heads.eval()
    probability_batches = {name: [] for name in MODALITIES}
    label_batches = []
    with torch.no_grad():
        for eeg, audio, video, labels in loader:
            inputs = (eeg.to(device), audio.to(device), video.to(device))
            labels = labels.to(device)
            logits = heads(model.encode_modalities(*inputs))
            for name in MODALITIES:
                probability_batches[name].append(logits[name].softmax(dim=1).cpu())
            label_batches.append(labels.cpu())

    all_labels = torch.cat(label_batches)
    accuracies = {}
    diagnostics = {}
    for name in MODALITIES:
        probabilities = torch.cat(probability_batches[name])
        confidence, prediction = probabilities.max(dim=1)
        correct = prediction.eq(all_labels)
        accuracies[name] = correct.float().mean().item()
        expected_calibration_error = 0.0
        for lower in torch.linspace(0.5, 0.95, 10):
            upper = lower + 0.05
            mask = (confidence >= lower) & (
                confidence <= upper if upper >= 1.0 else confidence < upper
            )
            if mask.any():
                bin_fraction = mask.float().mean()
                expected_calibration_error += (
                    bin_fraction
                    * (correct[mask].float().mean() - confidence[mask].mean()).abs()
                ).item()
        diagnostics[name] = {
            "accuracy": accuracies[name],
            "mean_confidence": confidence.mean().item(),
            "expected_calibration_error_10_bin": expected_calibration_error,
        }
    return accuracies, diagnostics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--baseline-checkpoint", default="best_multimodal_model.pth")
    parser.add_argument("--output", default="modality_heads.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    splits = load_binary_splits(args.data_root, seed=args.seed)
    train_loader = make_loader(splits["train"], args.batch_size, True, args.seed)
    validation_loader = make_loader(
        splits["validation"], args.batch_size, False, args.seed
    )

    model = EAVMultimodalFusionNetwork(num_classes=2, embedding_dim=128).to(device)
    model.load_state_dict(load_state_dict(args.baseline_checkpoint, device), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    heads = ModalityPredictionHeads(embedding_dim=128, num_classes=2).to(device)
    optimizer = torch.optim.Adam(
        heads.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    best_score = -1.0
    best_state = None
    best_accuracies = None
    best_confidence_diagnostics = None

    for epoch in range(args.epochs):
        heads.train()
        running_loss = total = 0
        for eeg, audio, video, labels in train_loader:
            inputs = (eeg.to(device), audio.to(device), video.to(device))
            labels = labels.to(device)
            with torch.no_grad():
                features = model.encode_modalities(*inputs)
            logits = heads(features)
            loss = sum(criterion(logits[name], labels) for name in MODALITIES) / 3.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            total += len(labels)

        accuracies, confidence_diagnostics = evaluate_heads(
            model, heads, validation_loader, device
        )
        score = sum(accuracies.values()) / len(accuracies)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs}: loss={running_loss / total:.4f}; "
            + "; ".join(f"{name} val_acc={value:.4f}" for name, value in accuracies.items())
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(heads.state_dict())
            best_accuracies = accuracies
            best_confidence_diagnostics = confidence_diagnostics

    denominator = sum(best_accuracies.values())
    fixed_weights = (
        {name: best_accuracies[name] / denominator for name in MODALITIES}
        if denominator > 0
        else {name: 1.0 / 3.0 for name in MODALITIES}
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "validation_accuracy": best_accuracies,
            "validation_confidence_diagnostics": best_confidence_diagnostics,
            "fixed_weights": fixed_weights,
            "seed": args.seed,
            "label_mapping": {0: "Sad", 1: "Happy"},
            "baseline_checkpoint": args.baseline_checkpoint,
        },
        args.output,
    )
    print(f"Saved {args.output}; validation-derived fixed weights={fixed_weights}")


if __name__ == "__main__":
    main()
