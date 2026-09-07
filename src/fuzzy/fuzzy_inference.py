"""Mamdani inference and centroid defuzzification for modality reliability."""

from typing import Dict, Iterable, Sequence

import numpy as np

from .fuzzy_rules import FuzzyRule, build_rule_base
from .membership_functions import (
    CONFIDENCE_PARTITION,
    DISAGREEMENT_PARTITION,
    WEIGHT_PARTITION,
)


MODALITIES = ("eeg", "audio", "video")


def _integrate(values, universe):
    """Use NumPy 2's name when available, with a NumPy 1.x fallback."""
    function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return function(values, universe)


def mean_pairwise_total_variation(probabilities: Sequence[Sequence[float]]) -> float:
    """Return normalized cross-modal disagreement in [0, 1].

    For distributions p_e, p_a, p_v:

        D = (TV(p_e,p_a) + TV(p_e,p_v) + TV(p_a,p_v)) / 3
        TV(p,q) = 0.5 * sum_c |p_c - q_c|

    Total variation is bounded by one for valid probability distributions, so
    their pairwise mean is also bounded by one.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (3, 2):
        raise ValueError(f"Expected probability shape (3, 2), received {probs.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("Each modality probability distribution must sum to one")
    distances = [
        0.5 * np.abs(probs[i] - probs[j]).sum()
        for i, j in ((0, 1), (0, 2), (1, 2))
    ]
    return float(np.mean(distances))


class MamdaniReliabilityFIS:
    """Four-input, three-output Mamdani fuzzy inference system."""

    def __init__(
        self,
        rules: Iterable[FuzzyRule] = None,
        universe_resolution: int = 1001,
    ):
        if universe_resolution < 101:
            raise ValueError("universe_resolution must be at least 101")
        self.rules = list(rules) if rules is not None else build_rule_base()
        self.output_universe = np.linspace(0.0, 1.0, universe_resolution)
        self.output_memberships = {
            level: WEIGHT_PARTITION.membership(level, self.output_universe)
            for level in ("low", "medium", "high")
        }

    @staticmethod
    def _memberships(inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        memberships = {}
        for name, value in inputs.items():
            partition = (
                DISAGREEMENT_PARTITION
                if name == "disagreement"
                else CONFIDENCE_PARTITION
            )
            memberships[name] = {
                level: partition.membership(level, value)
                for level in ("low", "medium", "high")
            }
        return memberships

    def infer(
        self,
        eeg_confidence: float,
        audio_confidence: float,
        video_confidence: float,
        disagreement: float,
    ) -> np.ndarray:
        inputs = {
            "eeg_confidence": float(eeg_confidence),
            "audio_confidence": float(audio_confidence),
            "video_confidence": float(video_confidence),
            "disagreement": float(disagreement),
        }
        memberships = self._memberships(inputs)
        aggregated = {
            modality: np.zeros_like(self.output_universe) for modality in MODALITIES
        }

        # Mamdani AND uses min; implication clips the consequent; rule outputs
        # and all rules are aggregated with max.
        for rule in self.rules:
            strength = min(
                memberships[input_name][level]
                for input_name, level in rule.antecedents.items()
            )
            if strength <= 0:
                continue
            for modality, output_level in rule.consequents.items():
                clipped = np.minimum(strength, self.output_memberships[output_level])
                aggregated[modality] = np.maximum(aggregated[modality], clipped)

        crisp = []
        for modality in MODALITIES:
            membership = aggregated[modality]
            area = _integrate(membership, self.output_universe)
            if area <= np.finfo(float).eps:
                crisp.append(1.0 / 3.0)
            else:
                moment = _integrate(
                    self.output_universe * membership, self.output_universe
                )
                crisp.append(float(moment / area))

        weights = np.asarray(crisp, dtype=float)
        denominator = weights.sum()  # Compute once before normalizing any weight.
        if denominator <= np.finfo(float).eps:
            return np.full(3, 1.0 / 3.0)
        return weights / denominator

    def weights_from_probabilities(
        self, probabilities: Sequence[Sequence[float]]
    ) -> np.ndarray:
        probs = np.asarray(probabilities, dtype=float)
        disagreement = mean_pairwise_total_variation(probs)
        confidences = probs.max(axis=1)
        return self.infer(*confidences, disagreement)
