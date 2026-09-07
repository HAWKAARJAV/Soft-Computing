"""Independent classifiers for the frozen EEG, audio, and video experts."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITIES = ("eeg", "audio", "video")


class ModalityPredictionHeads(nn.Module):
    """Apply one 2-class linear classifier to each 128-D representation.

    These heads intentionally live outside ``EAVMultimodalFusionNetwork`` so
    adding them does not change the key schema of best_multimodal_model.pth.
    """

    def __init__(self, embedding_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.eeg_head = nn.Linear(embedding_dim, num_classes)
        self.audio_head = nn.Linear(embedding_dim, num_classes)
        self.video_head = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        eeg, audio, video = features
        return {
            "eeg": self.eeg_head(eeg),
            "audio": self.audio_head(audio),
            "video": self.video_head(video),
        }

    def probabilities(
        self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            name: F.softmax(logits, dim=-1)
            for name, logits in self(features).items()
        }
