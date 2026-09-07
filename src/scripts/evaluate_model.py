"""Evaluate only the unchanged original multimodal classifier baseline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.data.splits import load_binary_splits
from src.models.multimodal_model import EAVMultimodalFusionNetwork
from src.scripts.experiment_utils import (
    CLASS_NAMES,
    load_state_dict,
    make_loader,
    seed_everything,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint", default="best_multimodal_model.pth")
    parser.add_argument("--output-dir", default="docs/images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_binary_splits(args.data_root, seed=args.seed)
    loader = make_loader(splits["test"], args.batch_size, False, args.seed)
    model = EAVMultimodalFusionNetwork(num_classes=2, embedding_dim=128).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    labels, predictions, positive_probabilities = [], [], []
    with torch.no_grad():
        for eeg, audio, video, batch_labels in loader:
            logits = model(eeg.to(device), audio.to(device), video.to(device))
            probabilities = F.softmax(logits, dim=1)
            labels.extend(batch_labels.numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            positive_probabilities.extend(probabilities[:, 1].cpu().numpy())

    print(
        classification_report(
            labels,
            predictions,
            labels=(0, 1),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(labels, predictions, labels=(0, 1))
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(title="Original Multimodal Baseline", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)

    try:
        area = roc_auc_score(labels, positive_probabilities)
        false_positive, true_positive, _ = roc_curve(labels, positive_probabilities)
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(false_positive, true_positive, label=f"ROC-AUC = {area:.3f}")
        axis.plot((0, 1), (0, 1), "--", color="grey")
        axis.set(
            title="Original Multimodal Baseline ROC",
            xlabel="False-positive rate",
            ylabel="True-positive rate",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close(figure)
        print(f"ROC-AUC: {area:.4f}")
    except ValueError:
        print("ROC-AUC unavailable because the test split contains one class.")


if __name__ == "__main__":
    main()
