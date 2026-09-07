"""Membership functions used by the reliability FIS.

The breakpoints form overlapping, interpretable partitions of each bounded
input domain. Confidence is restricted to [0.5, 1] because it is the maximum
of a binary softmax. Disagreement and reliability weights use [0, 1].
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


def triangular(x, left: float, peak: float, right: float):
    """Triangular membership supporting scalar or NumPy-array inputs."""
    values = np.asarray(x, dtype=float)
    result = np.zeros_like(values)
    if peak > left:
        result = np.maximum(result, np.minimum((values - left) / (peak - left), 1.0))
    else:
        result = np.maximum(result, (values <= peak).astype(float))
    if right > peak:
        result = np.minimum(result, np.maximum((right - values) / (right - peak), 0.0))
    else:
        result = np.minimum(result, (values <= right).astype(float))
    result = np.clip(result, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def trapezoidal(x, left: float, shoulder_left: float, shoulder_right: float, right: float):
    """Trapezoidal membership supporting vertical left/right shoulders."""
    values = np.asarray(x, dtype=float)
    if shoulder_left == left:
        rising = (values >= left).astype(float)
    else:
        rising = np.clip((values - left) / (shoulder_left - left), 0.0, 1.0)
    if right == shoulder_right:
        falling = (values <= right).astype(float)
    else:
        falling = np.clip((right - values) / (right - shoulder_right), 0.0, 1.0)
    result = np.where(
        (values < left) | (values > right),
        0.0,
        np.minimum(rising, falling),
    )
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class LinguisticPartition:
    domain_min: float
    domain_max: float
    kind_and_points: Dict[str, tuple]

    def membership(self, label: str, value):
        kind, *points = self.kind_and_points[label]
        clipped = np.clip(value, self.domain_min, self.domain_max)
        function = triangular if kind == "triangle" else trapezoidal
        return function(clipped, *points)


# Equal-width confidence regions centered at 0.50, 0.75, and 1.00.
CONFIDENCE_PARTITION = LinguisticPartition(
    0.5,
    1.0,
    {
        "low": ("trapezoid", 0.5, 0.5, 0.625, 0.75),
        "medium": ("triangle", 0.5, 0.75, 1.0),
        "high": ("trapezoid", 0.75, 0.875, 1.0, 1.0),
    },
)

# Standard overlapping quarter/half/three-quarter partition on [0, 1].
UNIT_PARTITION = LinguisticPartition(
    0.0,
    1.0,
    {
        "low": ("trapezoid", 0.0, 0.0, 0.25, 0.5),
        "medium": ("triangle", 0.25, 0.5, 0.75),
        "high": ("trapezoid", 0.5, 0.75, 1.0, 1.0),
    },
)

DISAGREEMENT_PARTITION = UNIT_PARTITION
WEIGHT_PARTITION = UNIT_PARTITION
