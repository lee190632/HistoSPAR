"""
Double-Tier Feature Distillation Multiple Instance Learning (DTFD-MIL).
Primary champion architecture powering patient-level clinical prediction in HistoSPAR.
Reference: Zhang et al., CVPR 2022.
"""
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """
    Computes attention weights via tanh and sigmoid gating (Ilse et al., ICML 2018):
        a_i = softmax(w^T [tanh(V h_i) * sigmoid(U h_i)])
    """
    def __init__(self, in_features: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (N, in_features) instance representations

        Returns:
            m: (1, in_features) attention-pooled bag representation
            A: (N, 1) normalized attention distribution
        """
        A_V = self.attention_V(h)  # (N, hidden_dim)
        A_U = self.attention_U(h)  # (N, hidden_dim)
        A = self.attention_w(A_V * A_U)  # (N, 1)
        A = F.softmax(A, dim=0)  # Attention distribution over instances
        m = (A * h).sum(dim=0, keepdim=True)  # Pooled representation (1, in_features)
        return m, A


class DTFD_MIL(nn.Module):
    """
    Double-Tier Feature Distillation Multiple Instance Learning (DTFD-MIL).
    Tier 1 aggregates pseudo-bags and optimizes instance distillation.
    Tier 2 aggregates distilled pseudo-bag vectors into the final patient prediction.
    """
    def __init__(self, feature_dim: int = 512, n_classes: int = 2) -> None:
        super().__init__()
        self.tier1_attention = GatedAttention(feature_dim, hidden_dim=128)
        self.tier1_classifier = nn.Linear(feature_dim, n_classes)

        self.tier2_attention = GatedAttention(feature_dim, hidden_dim=128)
        self.tier2_classifier = nn.Linear(feature_dim, n_classes)

    def forward(
        self,
        embeddings: torch.Tensor,
        num_pseudo_bags: int = 6,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (N, 512) regional fusion representations
            num_pseudo_bags: Target number of pseudo-bags to partition instances into
            is_training: If True, uses random shuffle; if False, uses deterministic ordering

        Returns:
            patient_logits: (1, n_classes) final Tier 2 patient classification logits
            tier1_logits: (num_pseudo_bags, n_classes) Tier 1 pseudo-bag classification logits
        """
        if is_training:
            indices = torch.randperm(embeddings.shape[0], device=embeddings.device)
        else:
            indices = torch.arange(embeddings.shape[0], device=embeddings.device)

        k_bags = min(num_pseudo_bags, max(1, embeddings.shape[0]))
        splits = torch.tensor_split(indices, k_bags)

        pseudo_bags: List[torch.Tensor] = []
        tier1_logits_list: List[torch.Tensor] = []
        for split in splits:
            sub_h = embeddings[split]
            m_sub, _ = self.tier1_attention(sub_h)
            pseudo_bags.append(m_sub)
            tier1_logits_list.append(self.tier1_classifier(m_sub))

        U = torch.cat(pseudo_bags, dim=0)  # (num_pseudo_bags, 512)
        tier1_logits = torch.cat(tier1_logits_list, dim=0)  # (num_pseudo_bags, n_classes)

        # Gradient detachment for feature distillation
        if is_training:
            U = U.detach()

        # Tier 2 patient-level aggregation and classification
        m_patient, _ = self.tier2_attention(U)
        patient_logits = self.tier2_classifier(m_patient)  # (1, n_classes)

        return patient_logits, tier1_logits
