"""Complete, deterministic rule base for adaptive reliability.

There is one rule for every combination of three confidence levels and one
disagreement level: 3^4 = 81 rules. This exhaustive grid prevents uncovered
regions and makes the policy inspectable instead of hiding numeric weights.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List


LEVELS = ("low", "medium", "high")
RANK = {level: index for index, level in enumerate(LEVELS)}


@dataclass(frozen=True)
class FuzzyRule:
    antecedents: Dict[str, str]
    consequents: Dict[str, str]

    def describe(self) -> str:
        conditions = " AND ".join(
            f"{name} is {level.upper()}" for name, level in self.antecedents.items()
        )
        outputs = ", ".join(
            f"{name} weight is {level.upper()}"
            for name, level in self.consequents.items()
        )
        return f"IF {conditions} THEN {outputs}"


def _consequents(confidences: Dict[str, str], disagreement: str) -> Dict[str, str]:
    """Translate linguistic confidence into reliability.

    Low disagreement preserves each modality's confidence level. Under medium
    disagreement, low-confidence modalities remain low while other modalities
    retain their evidence. Under high disagreement, a unique most-confident
    modality is retained and all alternatives are reduced by one level. If the
    highest confidence is tied, every tied leader is reduced one level because
    the four requested inputs cannot identify which tied expert is incorrect.
    """
    if disagreement != "high":
        return dict(confidences)

    ranks = {name: RANK[level] for name, level in confidences.items()}
    best_rank = max(ranks.values())
    leaders = [name for name, rank in ranks.items() if rank == best_rank]
    outputs = {}
    for name, rank in ranks.items():
        preserve_unique_leader = len(leaders) == 1 and name == leaders[0]
        output_rank = rank if preserve_unique_leader else max(0, rank - 1)
        outputs[name] = LEVELS[output_rank]
    return outputs


def build_rule_base() -> List[FuzzyRule]:
    rules = []
    modalities = ("eeg", "audio", "video")
    for eeg, audio, video, disagreement in product(LEVELS, repeat=4):
        confidences = {"eeg": eeg, "audio": audio, "video": video}
        rules.append(
            FuzzyRule(
                antecedents={
                    "eeg_confidence": eeg,
                    "audio_confidence": audio,
                    "video_confidence": video,
                    "disagreement": disagreement,
                },
                consequents=_consequents(confidences, disagreement),
            )
        )
    return rules
