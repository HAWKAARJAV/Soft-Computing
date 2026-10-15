"""Generate a small synthetic Sad/Happy EAV-style dataset for demos and coursework.

The tensors match the real preprocessed layout (EEG, audio, video) but embed
class-specific patterns plus subject noise so multimodal training reaches
realistic accuracy without downloading the full EAV corpus.

Output layout (compatible with load_subject_multimodal):

    <output-root>/Inputs/Multimodal/subject_01.npz
      keys: eeg, audio, video, labels
      labels use original EAV codes: 1=Sad, 3=Happy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

EEG_SHAPE = (30, 500)
AUDIO_LEN = 80000
VIDEO_SHAPE = (25, 3, 56, 56)
SAMPLE_RATE = 16000
DURATION_S = AUDIO_LEN / SAMPLE_RATE


def _eeg_wave(
    class_index: int,
    subject_id: int,
    rng: np.random.Generator,
    separability: float,
) -> np.ndarray:
    """class_index: 0=Sad, 1=Happy."""
    time = np.linspace(0.0, 1.0, EEG_SHAPE[1], dtype=np.float32)
    signal = np.zeros(EEG_SHAPE, dtype=np.float32)
    if class_index == 0:
        active = np.arange(0, 18)
        carrier = 7.5
        envelope = 0.85 + 0.15 * np.sin(2 * np.pi * 1.2 * time)
    else:
        active = np.arange(12, 30)
        carrier = 13.5
        envelope = 0.85 + 0.15 * np.sin(2 * np.pi * 2.1 * time + 0.4)

    strength = float(np.clip(separability, 0.2, 1.0))
    for channel in active:
        phase = rng.uniform(0.0, 2 * np.pi)
        signal[channel] = strength * envelope * np.sin(2 * np.pi * carrier * time + phase)

    inactive = np.setdiff1d(np.arange(EEG_SHAPE[0]), active)
    for channel in inactive:
        phase = rng.uniform(0.0, 2 * np.pi)
        signal[channel] = 0.35 * strength * np.sin(2 * np.pi * (carrier + 3.0) * time + phase)

    subject_shift = 0.12 * np.sin(2 * np.pi * (subject_id % 5 + 1) * time)
    signal += subject_shift[None, :]
    signal += rng.normal(0.0, 0.34, size=EEG_SHAPE).astype(np.float32)
    return signal


def _audio_wave(
    class_index: int,
    subject_id: int,
    rng: np.random.Generator,
    separability: float,
) -> np.ndarray:
    time = np.linspace(0.0, DURATION_S, AUDIO_LEN, dtype=np.float32)
    if class_index == 0:
        freqs = (165.0, 248.0)
        amplitudes = (0.9, 0.35)
    else:
        freqs = (285.0, 428.0)
        amplitudes = (0.85, 0.4)

    wave = np.zeros(AUDIO_LEN, dtype=np.float32)
    strength = float(np.clip(separability, 0.2, 1.0))
    for frequency, amplitude in zip(freqs, amplitudes):
        phase = rng.uniform(0.0, 2 * np.pi)
        wave += strength * amplitude * np.sin(2 * np.pi * frequency * time + phase)

    # Shared nuisance harmonics (both classes) make unimodal fusion less trivial.
    wave += 0.55 * np.sin(2 * np.pi * 210.0 * time + subject_id * 0.17)
    tremolo = 1.0 + 0.12 * np.sin(2 * np.pi * (1.7 + 0.1 * subject_id) * time)
    wave *= tremolo.astype(np.float32)
    wave += rng.normal(0.0, 0.26, size=AUDIO_LEN).astype(np.float32)
    return wave.astype(np.float32)


def _video_volume(
    class_index: int, rng: np.random.Generator, separability: float
) -> np.ndarray:
    frames = np.zeros(VIDEO_SHAPE, dtype=np.float32)
    if class_index == 0:
        base = np.array([0.22, 0.28, 0.42], dtype=np.float32)
        mouth_curve = -0.06
    else:
        base = np.array([0.48, 0.36, 0.28], dtype=np.float32)
        mouth_curve = 0.09

    strength = float(np.clip(separability, 0.2, 1.0))
    for frame_idx in range(VIDEO_SHAPE[0]):
        progress = frame_idx / max(VIDEO_SHAPE[0] - 1, 1)
        tint = base + strength * 0.05 * np.sin(2 * np.pi * progress)
        frames[frame_idx] = tint[:, None, None]
        center_y, center_x = 30, 28
        radius_y, radius_x = 10, 14
        yy = np.arange(56, dtype=np.float32)[:, None]
        xx = np.arange(56, dtype=np.float32)[None, :]
        mouth_y = center_y + int(mouth_curve * 20 * progress)
        mask = ((yy - mouth_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2 < 1.0
        frames[frame_idx, 0, mask] += 0.18
        frames[frame_idx, 1, mask] += 0.10

    frames += rng.normal(0.0, 0.09, size=VIDEO_SHAPE).astype(np.float32)
    return np.clip(frames, 0.0, 1.0)


def _apply_hardness(
    class_index: int,
    eeg: np.ndarray,
    audio: np.ndarray,
    video: np.ndarray,
    hardness: float,
    rng: np.random.Generator,
    separability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend one modality toward the opposite class for harder examples."""
    if hardness <= 0:
        return eeg, audio, video
    wrong = 1 - class_index
    modality = int(rng.integers(0, 3))
    blend = float(np.clip(hardness, 0.0, 1.0))
    if modality == 0:
        eeg = (1.0 - blend) * eeg + blend * _eeg_wave(wrong, 1, rng, separability)
    elif modality == 1:
        audio = (1.0 - blend) * audio + blend * _audio_wave(wrong, 1, rng, separability)
    else:
        video = (1.0 - blend) * video + blend * _video_volume(wrong, rng, separability)
    return eeg, audio, video


def generate_subject(
    subject_id: int,
    samples_per_class: int,
    hard_fraction: float,
    hardness: float,
    separability: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    eeg_list, audio_list, video_list, label_list = [], [], [], []
    for class_index, eav_label in ((0, 1), (1, 3)):
        for _ in range(samples_per_class):
            eeg = _eeg_wave(class_index, subject_id, rng, separability)
            audio = _audio_wave(class_index, subject_id, rng, separability)
            video = _video_volume(class_index, rng, separability)
            if rng.random() < hard_fraction:
                sample_hardness = hardness * rng.uniform(0.55, 1.0)
                eeg, audio, video = _apply_hardness(
                    class_index,
                    eeg,
                    audio,
                    video,
                    sample_hardness,
                    rng,
                    separability,
                )
            eeg_list.append(eeg)
            audio_list.append(audio)
            video_list.append(video)
            label_list.append(eav_label)

    return {
        "eeg": np.stack(eeg_list, axis=0),
        "audio": np.stack(audio_list, axis=0),
        "video": np.stack(video_list, axis=0),
        "labels": np.asarray(label_list, dtype=np.int64),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/synthetic",
        help="Root folder; NPZ files go under Inputs/Multimodal/",
    )
    parser.add_argument("--subjects", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=28)
    parser.add_argument(
        "--hard-fraction",
        type=float,
        default=0.24,
        help="Fraction of samples with one modality partially flipped.",
    )
    parser.add_argument(
        "--hardness",
        type=float,
        default=0.58,
        help="How strongly hard samples mix the wrong-class modality.",
    )
    parser.add_argument(
        "--separability",
        type=float,
        default=0.72,
        help="Class signal strength (lower -> more realistic overlap, ~85-92% acc).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_root) / "Inputs" / "Multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for subject_id in range(1, args.subjects + 1):
        arrays = generate_subject(
            subject_id,
            args.samples_per_class,
            args.hard_fraction,
            args.hardness,
            args.separability,
            rng,
        )
        path = out_dir / f"subject_{subject_id:02d}.npz"
        np.savez_compressed(path, **arrays)
        total_samples += len(arrays["labels"])
        print(
            f"Wrote {path.name}: {len(arrays['labels'])} samples "
            f"(Sad={(arrays['labels'] == 1).sum()}, "
            f"Happy={(arrays['labels'] == 3).sum()})"
        )

    readme = Path(args.output_root) / "README_SYNTHETIC.txt"
    readme.write_text(
        "Synthetic demo data for EAV-Emotion-Recognition.\n"
        "Not real EEG/audio/video — generated for coursework evaluation only.\n"
        f"Subjects: {args.subjects}, per-class samples: {args.samples_per_class}\n",
        encoding="utf-8",
    )
    print(f"\nDone. {args.subjects} subjects, {total_samples} total samples.")
    print(f"Use: python3 src/scripts/run_synthetic_demo.py --data-root {args.output_root}")


if __name__ == "__main__":
    main()
