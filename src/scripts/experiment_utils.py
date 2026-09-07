"""Shared deterministic utilities for EAV training and evaluation scripts."""

import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import EAVMultimodalDataset


CLASS_NAMES = ("Sad", "Happy")
MODALITIES = ("eeg", "audio", "video")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(split, batch_size: int, shuffle: bool, seed: int, drop_last: bool = False):
    dataset = EAVMultimodalDataset(
        split.eeg, split.audio, split.video, split.labels
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        num_workers=0,
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_state_dict(path, device):
    # weights_only prevents arbitrary pickle execution for ordinary state dicts.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location=device)
    return extract_state_dict(checkpoint)


def save_json(value: Dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
