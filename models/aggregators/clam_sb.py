"""
CLAM Single-Branch (CLAM-SB) with instance-level clustering constraint.
Comparative benchmark backbone in HistoSPAR.
Reference: Lu et al., Nature Biomedical Engineering 2021.
"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dtfd_mil import GatedAttention


class CLAM_SB(nn.Module):
    """Comparative Benchmark: CLAM single-branch with instance-level clustering constraint (Lu et al., 2021)."""
    def __init__(self, feature_dim: int = 512, n_classes: int = 2, k_sample: int = 8) -> None:
        super().__init__()
        self.attention = GatedAttention(feature_dim, hidden_dim=256)
        self.classifier = nn.Linear(feature_dim, n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(feature_dim, 2) for _ in range(n_classes)])
        self.k_sample = k_sample

    def forward(
        self,
        embeddings: torch.Tensor,
        label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m, A = self.attention(embeddings)
        bag_logits = self.classifier(m)

        # Instance-level clustering on top attended patches
        instance_loss = torch.tensor(0.0, device=embeddings.device)
        if label is not None and self.training:
            k = min(self.k_sample, embeddings.shape[0] // 2)
            if k > 0:
                _, top_p_idx = torch.topk(A.squeeze(-1), k, largest=True)
                _, top_n_idx = torch.topk(A.squeeze(-1), k, largest=False)
                p_preds = self.instance_classifiers[label](embeddings[top_p_idx])
                n_preds = self.instance_classifiers[label](embeddings[top_n_idx])
                target_p = torch.ones(k, dtype=torch.long, device=embeddings.device)
                target_n = torch.zeros(k, dtype=torch.long, device=embeddings.device)
                instance_loss = F.cross_entropy(p_preds, target_p) + F.cross_entropy(n_preds, target_n)

        return bag_logits, instance_loss
