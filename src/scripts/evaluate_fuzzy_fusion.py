"""Compare baseline, unimodal, simple, fixed, and fuzzy fusion on held-out test data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.splits import load_binary_splits
from src.models.fuzzy_fusion import FuzzyProbabilityFusion
from src.models.modality_heads import MODALITIES, ModalityPredictionHeads
from src.models.multimodal_model import EAVMultimodalFusionNetwork
from src.scripts.experiment_utils import (
    CLASS_NAMES,
    load_state_dict,
    make_loader,
    save_json,
    seed_everything,
    select_device,
)


METHODS = (
    "original_multimodal",
    "eeg_only",
    "audio_only",
    "video_only",
    "average_fusion",
    "fixed_weighted_fusion",
    "fuzzy_adaptive_fusion",
)
ROBUSTNESS_CONDITIONS = (
    "normal",
    "poor_audio",
    "poor_video",
    "poor_eeg",
    "drop_audio",
    "drop_video",
    "drop_eeg",
    "audio_disagreement",
)


def load_heads_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def perturb_inputs(eeg, audio, video, condition, generator, noise_scale):
    eeg, audio, video = eeg.clone(), audio.clone(), video.clone()
    if condition in ("poor_eeg", "poor_audio"):
        target = eeg if condition == "poor_eeg" else audio
        dimensions = tuple(range(1, target.ndim))
        scale = target.std(dim=dimensions, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype)
        target.add_(noise_scale * scale * noise)
    elif condition == "poor_video":
        batch, frames, channels, height, width = video.shape
        flattened = video.reshape(batch * frames, channels, height, width)
        degraded = F.interpolate(flattened, size=(14, 14), mode="area")
        degraded = F.interpolate(
            degraded, size=(height, width), mode="bilinear", align_corners=False
        )
        video = degraded.reshape_as(video)
    elif condition == "drop_eeg":
        eeg.zero_()
    elif condition == "drop_audio":
        audio.zero_()
    elif condition == "drop_video":
        video.zero_()
    return eeg, audio, video


def collect_probabilities(model, heads, loader, device, condition, seed, noise_scale):
    labels = []
    baseline_probabilities = []
    modality_probabilities = {name: [] for name in MODALITIES}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    heads.eval()
    with torch.no_grad():
        for eeg, audio, video, batch_labels in loader:
            eeg, audio, video = perturb_inputs(
                eeg, audio, video, condition, generator, noise_scale
            )
            inputs = (eeg.to(device), audio.to(device), video.to(device))
            baseline_logits, features = model.forward_with_features(*inputs)
            head_logits = heads(features)
            baseline_probabilities.append(
                F.softmax(baseline_logits, dim=1).cpu().numpy()
            )
            for name in MODALITIES:
                modality_probabilities[name].append(
                    F.softmax(head_logits[name], dim=1).cpu().numpy()
                )
            labels.append(batch_labels.numpy())

    modality_probabilities = {
        name: np.concatenate(parts) for name, parts in modality_probabilities.items()
    }
    if condition == "audio_disagreement":
        # Controlled expert-level contradiction: preserve confidence while
        # reversing Audio's Sad/Happy distribution.
        modality_probabilities["audio"] = modality_probabilities["audio"][:, ::-1]
    return (
        np.concatenate(labels),
        np.concatenate(baseline_probabilities),
        modality_probabilities,
    )


def fuse_all(baseline, modality_probs, fixed_weights, fuzzy_fusion):
    stacked = np.stack([modality_probs[name] for name in MODALITIES], axis=1)
    average = stacked.mean(axis=1)
    fixed = sum(
        fixed_weights[name] * modality_probs[name] for name in MODALITIES
    )
    fuzzy, weights, disagreements = fuzzy_fusion.fuse_numpy(modality_probs)
    probabilities = {
        "original_multimodal": baseline,
        "eeg_only": modality_probs["eeg"],
        "audio_only": modality_probs["audio"],
        "video_only": modality_probs["video"],
        "average_fusion": average,
        "fixed_weighted_fusion": fixed,
        "fuzzy_adaptive_fusion": fuzzy,
    }
    return probabilities, weights, disagreements


def calculate_metrics(labels, probabilities):
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=(0, 1), zero_division=0
    )
    macro = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(labels, probabilities[:, 1])
    except ValueError:
        roc_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "roc_auc": float(roc_auc),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(2)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=(0, 1)).tolist(),
    }


def plot_normal_confusions(results, output_dir):
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, method in zip(axes.flat, METHODS):
        matrix = results["normal"]["metrics"][method]["confusion_matrix"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=axis,
        )
        axis.set_title(method.replace("_", " ").title())
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    axes.flat[-1].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices.png", dpi=200)
    plt.close(figure)


def plot_robustness(results, output_dir):
    conditions = list(results)
    figure, axes = plt.subplots(2, 1, figsize=(13, 9))
    x = np.arange(len(conditions))
    for method in ("original_multimodal", "average_fusion", "fixed_weighted_fusion",
                   "fuzzy_adaptive_fusion"):
        axes[0].plot(
            x,
            [results[c]["metrics"][method]["accuracy"] for c in conditions],
            marker="o",
            label=method.replace("_", " "),
        )
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x, conditions, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()

    width = 0.25
    for index, modality in enumerate(MODALITIES):
        axes[1].bar(
            x + (index - 1) * width,
            [results[c]["mean_fuzzy_weights"][modality] for c in conditions],
            width,
            label=modality,
        )
    axes[1].set_ylabel("Mean fuzzy reliability weight")
    axes[1].set_xticks(x, conditions, rotation=25, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "robustness_summary.png", dpi=200)
    plt.close(figure)


def save_weight_examples(labels, modality_probs, weights, disagreements, output_dir):
    confidences = np.stack(
        [modality_probs[name].max(axis=1) for name in MODALITIES], axis=1
    )
    selected = [0]
    for modality_index in range(3):
        selected.extend(
            [int(weights[:, modality_index].argmin()), int(weights[:, modality_index].argmax())]
        )
    selected = list(dict.fromkeys(selected))[:7]
    examples = []
    for index in selected:
        examples.append(
            {
                "sample_index": index,
                "true_label": CLASS_NAMES[int(labels[index])],
                "confidences": {
                    name: float(confidences[index, i]) for i, name in enumerate(MODALITIES)
                },
                "disagreement": float(disagreements[index]),
                "fuzzy_weights": {
                    name: float(weights[index, i]) for i, name in enumerate(MODALITIES)
                },
            }
        )
    save_json({"examples": examples}, output_dir / "fuzzy_weight_examples.json")

    figure, axes = plt.subplots(len(examples), 1, figsize=(9, 2.3 * len(examples)))
    axes = np.atleast_1d(axes)
    for axis, example in zip(axes, examples):
        values = [example["fuzzy_weights"][name] for name in MODALITIES]
        axis.bar(MODALITIES, values, color=("royalblue", "darkorange", "seagreen"))
        axis.set_ylim(0, 1)
        axis.set_ylabel("Weight")
        axis.set_title(
            f"Sample {example['sample_index']} ({example['true_label']}), "
            f"disagreement={example['disagreement']:.3f}"
        )
    figure.tight_layout()
    figure.savefig(output_dir / "fuzzy_weight_examples.png", dpi=200)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--baseline-checkpoint", default="best_multimodal_model.pth")
    parser.add_argument("--heads-checkpoint", default="modality_heads.pth")
    parser.add_argument("--output-dir", default="docs/fuzzy_results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Evaluate input degradation, missing modalities, and disagreement.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_binary_splits(args.data_root, seed=args.seed)
    test_loader = make_loader(splits["test"], args.batch_size, False, args.seed)

    model = EAVMultimodalFusionNetwork(num_classes=2, embedding_dim=128).to(device)
    model.load_state_dict(load_state_dict(args.baseline_checkpoint, device), strict=True)
    heads_checkpoint = load_heads_checkpoint(args.heads_checkpoint, device)
    heads = ModalityPredictionHeads(128, 2).to(device)
    heads.load_state_dict(heads_checkpoint["model_state_dict"], strict=True)
    fixed_weights = heads_checkpoint.get(
        "fixed_weights", {name: 1.0 / 3.0 for name in MODALITIES}
    )
    fixed_denominator = sum(float(fixed_weights[name]) for name in MODALITIES)
    fixed_weights = {
        name: float(fixed_weights[name]) / fixed_denominator for name in MODALITIES
    }

    conditions = ROBUSTNESS_CONDITIONS if args.robustness else ("normal",)
    fuzzy_fusion = FuzzyProbabilityFusion()
    results = {}
    normal_details = None
    for condition in conditions:
        labels, baseline, modality_probs = collect_probabilities(
            model, heads, test_loader, device, condition, args.seed, args.noise_scale
        )
        all_probabilities, weights, disagreements = fuse_all(
            baseline, modality_probs, fixed_weights, fuzzy_fusion
        )
        results[condition] = {
            "metrics": {
                method: calculate_metrics(labels, all_probabilities[method])
                for method in METHODS
            },
            "mean_fuzzy_weights": {
                name: float(weights[:, index].mean())
                for index, name in enumerate(MODALITIES)
            },
            "mean_disagreement": float(disagreements.mean()),
        }
        print(
            f"{condition}: "
            + ", ".join(
                f"{method}={results[condition]['metrics'][method]['accuracy']:.4f}"
                for method in METHODS
            )
        )
        if condition == "normal":
            normal_details = (labels, modality_probs, weights, disagreements)

    serializable = {
        "label_mapping": {"0": "Sad", "1": "Happy"},
        "fixed_weights_from_validation": fixed_weights,
        "disagreement_formula": (
            "mean of three pairwise total-variation distances; "
            "TV(p,q)=0.5*sum(abs(p-q))"
        ),
        "results": results,
    }
    save_json(serializable, output_dir / "metrics.json")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("condition", "method", "accuracy", "precision_macro",
             "recall_macro", "f1_macro", "roc_auc")
        )
        for condition, condition_results in results.items():
            for method, metrics in condition_results["metrics"].items():
                writer.writerow(
                    [condition, method]
                    + [metrics[name] for name in (
                        "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"
                    )]
                )

    plot_normal_confusions(results, output_dir)
    save_weight_examples(*normal_details, output_dir)
    if args.robustness:
        plot_robustness(results, output_dir)
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
