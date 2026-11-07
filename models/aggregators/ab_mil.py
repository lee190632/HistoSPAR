"""
Single-tier Attention-based Multiple Instance Learning (AB-MIL).
Comparative benchmark backbone in HistoSPAR.
Reference: Ilse et al., ICML 2018.
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from .dtfd_mil import GatedAttention


class AB_MIL(nn.Module):
    """Comparative Benchmark: Single-tier Attention-based Multiple Instance Learning (Ilse et al., 2018)."""
    def __init__(self, feature_dim: int = 512, n_classes: int = 2) -> None:
        super().__init__()
        self.attention = GatedAttention(feature_dim, hidden_dim=256)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m, A = self.attention(embeddings)
        logits = self.classifier(m)
        return logits, A
