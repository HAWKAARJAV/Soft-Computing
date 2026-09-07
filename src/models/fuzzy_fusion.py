"""Probability-level adaptive fusion built on the Mamdani reliability FIS."""

from typing import Dict, Tuple

import numpy as np
import torch

from src.fuzzy.fuzzy_inference import (
    MODALITIES,
    MamdaniReliabilityFIS,
    mean_pairwise_total_variation,
)


class FuzzyProbabilityFusion:
    """Fuse batches of expert probabilities without changing neural encoders."""

    def __init__(self, fis: MamdaniReliabilityFIS = None):
        self.fis = fis or MamdaniReliabilityFIS()

    def fuse_numpy(
        self, modality_probabilities: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = [np.asarray(modality_probabilities[name]) for name in MODALITIES]
        if not all(array.shape == arrays[0].shape for array in arrays):
            raise ValueError("All modality probability arrays must share a shape")
        if arrays[0].ndim != 2 or arrays[0].shape[1] != 2:
            raise ValueError("Expected each modality probability array to have shape (N, 2)")

        sample_probabilities = np.stack(arrays, axis=1)  # (N, 3, 2)
        weights = np.stack(
            [self.fis.weights_from_probabilities(sample) for sample in sample_probabilities]
        )
        disagreements = np.asarray(
            [mean_pairwise_total_variation(sample) for sample in sample_probabilities]
        )
        fused = (weights[:, :, None] * sample_probabilities).sum(axis=1)
        return fused, weights, disagreements

    def __call__(
        self, modality_probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = modality_probabilities["eeg"]
        numpy_probabilities = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in modality_probabilities.items()
        }
        fused, weights, disagreements = self.fuse_numpy(numpy_probabilities)
        return (
            torch.as_tensor(fused, device=reference.device, dtype=reference.dtype),
            torch.as_tensor(weights, device=reference.device, dtype=reference.dtype),
            torch.as_tensor(
                disagreements, device=reference.device, dtype=reference.dtype
            ),
        )
