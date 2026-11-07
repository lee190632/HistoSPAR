"""
Multimodal adaptive fusion modules for HistoSPAR.
Integrates regional morphological visual representations with predicted spatial pathway activities.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    """
    Learns patch-level gating weights between 1536-d morphology and 30-d pathway scores.
    Projects both modalities into a unified 512-dimensional embedding space.
    """
    def __init__(
        self,
        image_dim: int = 1536,
        pathway_dim: int = 30,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        # 2-way softmax gating network over concatenated patch features
        self.gate = nn.Sequential(
            nn.Linear(image_dim + pathway_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        # Modality-specific projection layers
        self.proj_image = nn.Linear(image_dim, hidden_dim)
        self.proj_pathway = nn.Linear(pathway_dim, hidden_dim)

        # Post-fusion non-linear transformation
        self.post_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

    def forward(self, F_img: torch.Tensor, Q_path: torch.Tensor) -> torch.Tensor:
        """
        F_img: (N, 1536) visual representations
        Q_path: (N, 30) predicted pathway scores
        """
        concat_features = torch.cat([F_img, Q_path], dim=-1)  # (N, 1566)
        weights = F.softmax(self.gate(concat_features), dim=-1)  # (N, 2)

        # Gated additive combination of projected features
        h_img = self.proj_image(F_img)
        h_path = self.proj_pathway(Q_path)
        h_fused = weights[:, 0:1] * h_img + weights[:, 1:2] * h_path

        return self.post_fusion(h_fused)  # (N, 512)


class PatchEncoder(nn.Module):
    """Encodes patch features for Fusion, Image-alone, or Pathway-alone experimental arms."""
    def __init__(
        self,
        arm: str = "Fusion",
        image_dim: int = 1536,
        pathway_dim: int = 30,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.arm = arm
        self.image_dim = image_dim
        self.pathway_dim = pathway_dim

        if arm == "Fusion":
            self.fusion_module = AdaptiveFusion(image_dim, pathway_dim, hidden_dim)
        elif arm == "Image":
            self.image_module = nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            )
        elif arm == "Pathway":
            self.pathway_module = nn.Sequential(
                nn.Linear(pathway_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.25),
            )

    def forward(
        self,
        bag: torch.Tensor,
        pathway_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for patch encoding.
        If pathway_scores is provided separately, bag is treated as image features.
        Otherwise, bag is expected to contain concatenated [F_img, Q_path] or modality features.
        """
        if self.arm == "Fusion":
            if pathway_scores is not None:
                return self.fusion_module(bag, pathway_scores)
            return self.fusion_module(bag[:, :self.image_dim], bag[:, self.image_dim:self.image_dim + self.pathway_dim])
        elif self.arm == "Image":
            return self.image_module(bag[:, :self.image_dim])
        elif self.arm == "Pathway":
            if pathway_scores is not None:
                return self.pathway_module(pathway_scores)
            if bag.shape[-1] > self.pathway_dim:
                return self.pathway_module(bag[:, self.image_dim:self.image_dim + self.pathway_dim])
            return self.pathway_module(bag)
