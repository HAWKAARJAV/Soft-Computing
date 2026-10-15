#!/usr/bin/env python3
"""One-shot demo: synthetic data -> train -> test metrics (for class evaluation)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str], cwd: Path = ROOT):
    print("\n>>", " ".join(command))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/synthetic")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=22)
    parser.add_argument("--head-epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        default="demo_checkpoints",
        help="Saved weights (not gitignored; safe to show teachers).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpt = checkpoint_dir / "synthetic_multimodal.pth"
    heads_ckpt = checkpoint_dir / "synthetic_modality_heads.pth"
    results_dir = Path("docs/synthetic_demo_results")

    python = sys.executable

    if not args.skip_generate:
        run(
            [
                python,
                "src/scripts/generate_synthetic_data.py",
                "--output-root",
                args.data_root,
                "--subjects",
                str(args.subjects),
                "--samples-per-class",
                str(args.samples_per_class),
                "--seed",
                str(args.seed),
            ]
        )

    run(
        [
            python,
            "src/scripts/train_multimodal.py",
            "--data-root",
            args.data_root,
            "--output",
            str(baseline_ckpt),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
    )

    run(
        [
            python,
            "src/scripts/evaluate_model.py",
            "--data-root",
            args.data_root,
            "--checkpoint",
            str(baseline_ckpt),
            "--output-dir",
            str(results_dir / "images"),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
    )

    run(
        [
            python,
            "src/scripts/train_modality_heads.py",
            "--data-root",
            args.data_root,
            "--baseline-checkpoint",
            str(baseline_ckpt),
            "--output",
            str(heads_ckpt),
            "--epochs",
            str(args.head_epochs),
            "--batch-size",
            str(max(args.batch_size, 8)),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
    )

    run(
        [
            python,
            "src/scripts/evaluate_fuzzy_fusion.py",
            "--data-root",
            args.data_root,
            "--baseline-checkpoint",
            str(baseline_ckpt),
            "--heads-checkpoint",
            str(heads_ckpt),
            "--output-dir",
            str(results_dir),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
    )

    metrics_path = results_dir / "metrics.json"
    summary = results_dir / "summary_for_demo.txt"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        normal = metrics["results"]["normal"]["metrics"]
        lines = [
            "Synthetic EAV demo — held-out TEST accuracy (Sad vs Happy)",
            f"Data folder: {args.data_root}",
            f"Baseline checkpoint: {baseline_ckpt}",
            "",
        ]
        for method, values in sorted(normal.items(), key=lambda item: -item[1]["accuracy"]):
            lines.append(
                f"  {method:28s}  accuracy={values['accuracy']:.1%}  "
                f"F1={values['f1_macro']:.3f}  AUC={values['roc_auc']:.3f}"
            )
        lines.append("")
        lines.append(
            "Note: tensors are procedurally generated (not real EAV recordings) "
            "for coursework demonstration."
        )
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if summary.exists():
        print("\n" + "=" * 72)
        print("DEMO SUMMARY (show this in evaluation)")
        print("=" * 72)
        print(summary.read_text(encoding="utf-8"))

    print("\nFinished. Key files:")
    print(f"  Checkpoint: {baseline_ckpt}")
    print(f"  Plots:      {results_dir / 'images'}")
    print(f"  Metrics:    {results_dir}")


if __name__ == "__main__":
    main()
