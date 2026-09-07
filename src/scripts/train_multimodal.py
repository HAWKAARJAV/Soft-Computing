"""Train the unchanged late-concatenation baseline without test leakage."""

import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.splits import load_binary_splits
from src.models.multimodal_model import EAVMultimodalFusionNetwork
from src.scripts.experiment_utils import (
    make_loader,
    seed_everything,
    select_device,
)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for eeg, audio, video, labels in loader:
            eeg, audio, video, labels = (
                eeg.to(device),
                audio.to(device),
                video.to(device),
                labels.to(device),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(eeg, audio, video)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="best_multimodal_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    splits = load_binary_splits(args.data_root, seed=args.seed)
    train_loader = make_loader(
        splits["train"], args.batch_size, True, args.seed, drop_last=True
    )
    validation_loader = make_loader(
        splits["validation"], args.batch_size, False, args.seed
    )
    print(
        f"Device={device}; train={len(train_loader.dataset)}; "
        f"validation={len(validation_loader.dataset)}; "
        f"test held out={len(splits['test'].labels)}"
    )

    model = EAVMultimodalFusionNetwork(num_classes=2, embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_accuracy = -1.0
    best_state = None
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs}: "
            f"train loss={train_loss:.4f} accuracy={train_accuracy:.4f}; "
            f"validation loss={validation_loss:.4f} "
            f"accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())

    # Save only the historical state_dict schema for backward compatibility.
    torch.save(best_state, args.output)
    print(f"Saved {args.output}; best validation accuracy={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
