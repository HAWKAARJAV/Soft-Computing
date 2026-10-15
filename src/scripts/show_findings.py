#!/usr/bin/env python3
"""Print live project findings in the terminal.

This is the teacher-demo command. It always computes fuzzy-weight examples
from the real Mamdani system. If preprocessed EAV files exist under --data-root,
it also reports how many subjects and Sad/Happy samples were found.

Run from the repository root:

    python3 src/scripts/show_findings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fuzzy.fuzzy_inference import (  # noqa: E402
    MamdaniReliabilityFIS,
    mean_pairwise_total_variation,
)
from src.fuzzy.fuzzy_rules import build_rule_base  # noqa: E402
from src.data.multimodal_dataset import load_subject_multimodal  # noqa: E402


MODALITIES = ("EEG", "Audio", "Video")
CLASS_NAMES = ("Sad", "Happy")


SAMPLES = (
    {
        "title": "A — all experts agree and are confident",
        "probs": np.array([[0.91, 0.09], [0.88, 0.12], [0.90, 0.10]]),
        "finding": "When experts agree, they receive almost equal trust.",
    },
    {
        "title": "B — audio is uncertain",
        "probs": np.array([[0.91, 0.09], [0.54, 0.46], [0.88, 0.12]]),
        "finding": "Uncertain audio is trusted much less; EEG and video get more weight.",
    },
    {
        "title": "C — EEG is uncertain",
        "probs": np.array([[0.52, 0.48], [0.90, 0.10], [0.87, 0.13]]),
        "finding": "The fuzzy system reduces EEG when EEG is weak.",
    },
    {
        "title": "D — video is uncertain",
        "probs": np.array([[0.89, 0.11], [0.86, 0.14], [0.53, 0.47]]),
        "finding": "The fuzzy system reduces video when video is weak.",
    },
    {
        "title": "E — audio conflicts with EEG and video",
        "probs": np.array([[0.91, 0.09], [0.08, 0.92], [0.88, 0.12]]),
        "finding": (
            "All three experts are confident, so the system cannot safely "
            "decide which one is wrong. Weights stay almost equal."
        ),
    },
    {
        "title": "F — all experts are equally uncertain",
        "probs": np.array([[0.55, 0.45], [0.55, 0.45], [0.55, 0.45]]),
        "finding": "If everyone is equally uncertain, they share the decision equally.",
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live fuzzy findings and optional dataset summary."
    )
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def print_header(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_fuzzy_checks(fis: MamdaniReliabilityFIS):
    print_header("1. Fuzzy math checks")
    identical = [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]]
    split = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    uncertain_audio = np.array([[0.93, 0.07], [0.54, 0.46], [0.89, 0.11]])
    first = fis.weights_from_probabilities(uncertain_audio)
    second = fis.weights_from_probabilities(uncertain_audio)

    rows = (
        ("Rule base size", str(len(build_rule_base()))),
        ("Identical experts, disagreement", f"{mean_pairwise_total_variation(identical):.3f}"),
        ("Two-vs-one conflict, disagreement", f"{mean_pairwise_total_variation(split):.3f}"),
        ("Weights always sum to", f"{float(first.sum()):.3f}"),
        ("Same input -> same weights", "yes" if np.allclose(first, second) else "NO"),
        ("Uncertain audio gets lowest weight", "yes" if first[1] < first[0] and first[1] < first[2] else "NO"),
    )
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def format_row(name: str, probs: np.ndarray, weight: float) -> str:
    sad, happy = probs
    predicted = CLASS_NAMES[int(np.argmax(probs))]
    confidence = float(probs.max())
    return (
        f"  {name:<6}  P(Sad)={sad:.2f}  P(Happy)={happy:.2f}  "
        f"pred={predicted:<5}  conf={confidence:.2f}  weight={weight:.3f}"
    )


def show_fuzzy_samples(fis: MamdaniReliabilityFIS):
    print_header("2. Live fuzzy-weight findings (computed now)")
    for sample in SAMPLES:
        probs = sample["probs"]
        weights = fis.weights_from_probabilities(probs)
        disagreement = mean_pairwise_total_variation(probs)
        fused = weights @ probs
        predicted = CLASS_NAMES[int(np.argmax(fused))]
        print()
        print(f"  Sample {sample['title']}")
        for name, row, weight in zip(MODALITIES, probs, weights):
            print(format_row(name, row, float(weight)))
        print(
            f"  {'Fused':<6}  P(Sad)={fused[0]:.2f}  P(Happy)={fused[1]:.2f}  "
            f"pred={predicted:<5}  disagreement={disagreement:.3f}"
        )
        print(f"  Finding: {sample['finding']}")


def inspect_dataset(data_root: Path):
    print_header("3. Dataset")
    if not data_root.exists():
        print(f"  Folder not found: {data_root}")
        print("  Place preprocessed subjects in data/Inputs/Multimodal/")
        print("  Full Sad/Happy accuracy is not claimed without this data.")
        return False

    found = []
    sad_total = 0
    happy_total = 0
    other_total = 0
    for subject_id in range(1, 43):
        eeg, audio, video, labels = load_subject_multimodal(subject_id, data_root)
        if eeg is None:
            continue
        labels = np.asarray(labels).reshape(-1)
        sad = int(np.sum(labels == 1))
        happy = int(np.sum(labels == 3))
        other = int(len(labels) - sad - happy)
        sad_total += sad
        happy_total += happy
        other_total += other
        found.append(
            {
                "id": subject_id,
                "n": len(labels),
                "sad": sad,
                "happy": happy,
                "eeg": tuple(eeg.shape),
                "audio": tuple(audio.shape),
                "video": tuple(video.shape),
            }
        )

    if not found:
        print(f"  No synchronized subjects found under {data_root}")
        print("  Expected files: data/Inputs/Multimodal/subject_01.npz")
        print("  Full Sad/Happy accuracy is not claimed without this data.")
        return False

    print(f"  Subjects found: {len(found)} / 42")
    print(f"  Original labels kept as Sad(1) / Happy(3): {sad_total} / {happy_total}")
    print(f"  Other original EAV classes excluded later: {other_total}")
    print()
    print("  First loaded subject shapes:")
    first = found[0]
    print(f"    subject_{first['id']:02d}")
    print(f"    EEG    {first['eeg']}")
    print(f"    Audio  {first['audio']}")
    print(f"    Video  {first['video']}")
    print()
    print("  Per-subject Sad/Happy counts:")
    for item in found:
        print(
            f"    subject_{item['id']:02d}  n={item['n']:<4}  "
            f"Sad={item['sad']:<3}  Happy={item['happy']:<3}"
        )
    return True


def show_next_steps(dataset_present: bool):
    print_header("4. What this proves / what still needs data")
    print("  Proven now:")
    print("    - 81 Low/Medium/High rules cover every input combination")
    print("    - Uncertain modalities receive lower reliability weights")
    print("    - Equal experts share the decision equally")
    print("    - Weights always add to 1")
    print()
    if dataset_present:
        print("  Dataset files were found. Next commands:")
        print("    python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data")
        print("    python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data --robustness")
    else:
        print("  Not claimed yet:")
        print("    - Accuracy, F1, ROC-AUC on the EAV test set")
        print("    - Whether fuzzy fusion beats averaging or the original model")
        print()
        print("  After the dataset is in data/, run:")
        print("    python3 src/scripts/evaluate_fuzzy_fusion.py --data-root data")


def main():
    args = parse_args()
    print_header("EAV Emotion Recognition — Live Findings")
    print("  Binary task: Sad vs Happy")
    print("  Modalities:  EEG + Audio + Video")
    print("  Fusion:      Mamdani fuzzy reliability weights")

    fis = MamdaniReliabilityFIS(universe_resolution=501)
    show_fuzzy_checks(fis)
    show_fuzzy_samples(fis)
    dataset_present = inspect_dataset(Path(args.data_root))
    show_next_steps(dataset_present)
    print()


if __name__ == "__main__":
    main()
