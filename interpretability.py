"""
Stage 4: Model Interpretability and Feature Attribution for HistoSPAR.
Implements Conditional Pathway Integrated Gradients (Conditional IG) along Gauss-Legendre quadrature paths
and Grouped Shapley Attribution across 31 players (morphology block + 30 pathway channels),
grounded exclusively on the champion DTFD-MIL Fusion framework.

Usage:
    python interpretability.py --channel 0 --steps 64
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import PatchEncoder
from models.aggregators import DTFD_MIL


class PatientResponseModel(nn.Module):
    """
    End-to-end wrapper combining PatchEncoder (AdaptiveFusion) and DTFD-MIL
    for gradient-based attribution and prospective patient-level inference.
    """
    def __init__(self, encoder: PatchEncoder, mil_model: DTFD_MIL) -> None:
        super().__init__()
        self.encoder = encoder
        self.mil = mil_model

    def forward(
        self,
        image_features: torch.Tensor,
        pathway_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for backpropagation.
        image_features: (N, 1536)
        pathway_scores: (N, 30)
        """
        h = self.encoder(image_features, pathway_scores)
        logits, _ = self.mil(h, num_pseudo_bags=4, is_training=False)
        return logits

    def predict_probability(self, bag: torch.Tensor) -> float:
        """Returns patient-level pCR probability."""
        self.eval()
        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            h = self.encoder(bag.to(device))
            logits, _ = self.mil(h, num_pseudo_bags=4, is_training=False)
            return float(F.softmax(logits, dim=-1)[0, 1].item())


def conditional_pathway_ig(
    dtfd_fusion_model: PatientResponseModel,
    image_features: torch.Tensor,
    pathway_scores: torch.Tensor,
    target_channel_k: int,
    steps: int = 64,
) -> Tuple[float, torch.Tensor]:
    """
    Computes Integrated Gradients for pathway channel k in the DTFD-MIL Fusion model
    while holding image features F and all other 29 pathway channels fixed.

    Args:
        dtfd_fusion_model: Fitted PatientResponseModel (AdaptiveFusion + DTFD_MIL).
        image_features: (N, 1536) regional histology visual representations.
        pathway_scores: (N, 30) reconstructed regional pathway enrichment scores.
        target_channel_k: Pathway channel index (0 to 29) to attribute.
        steps: Number of Gauss-Legendre quadrature integration steps.

    Returns:
        patient_pathway_attribution: Patient-level scalar attribution score.
        attribution_per_patch: (N,) patch-level attribution distribution.
    """
    dtfd_fusion_model.eval()

    # Baseline reference: target pathway set to 0.0 (standardized mean), others intact
    baseline = pathway_scores.clone()
    baseline[:, target_channel_k] = 0.0

    # Gauss-Legendre quadrature points and weights along the interpolation path [0, 1]
    alphas_np, weights_np = np.polynomial.legendre.leggauss(steps)
    alphas = torch.tensor(0.5 * (alphas_np + 1.0), dtype=torch.float32, device=pathway_scores.device)
    weights = torch.tensor(0.5 * weights_np, dtype=torch.float32, device=pathway_scores.device)

    total_grad = torch.zeros_like(pathway_scores[:, target_channel_k])

    for alpha, w in zip(alphas, weights):
        interpolated = baseline.clone()
        # Interpolate only the target channel k
        interpolated[:, target_channel_k] = baseline[:, target_channel_k] + alpha * (
            pathway_scores[:, target_channel_k] - baseline[:, target_channel_k]
        )
        interpolated.requires_grad_(True)

        # Objective: margin logit between pCR (class 1) and non-pCR (class 0) from DTFD-MIL Tier 2
        logits = dtfd_fusion_model(image_features, interpolated)
        margin = logits[0, 1] - logits[0, 0]
        margin.backward()

        if interpolated.grad is not None:
            grad_k = interpolated.grad[:, target_channel_k]
            total_grad += w * grad_k

    # Path integral scaled by the difference vector: (x_k - x'_k) * integral
    attribution_per_patch = (pathway_scores[:, target_channel_k] - 0.0) * total_grad
    patient_pathway_attribution = float(attribution_per_patch.sum().item())
    return patient_pathway_attribution, attribution_per_patch.detach()


def grouped_shapley_attribution(
    dtfd_fusion_model: PatientResponseModel,
    patient_bag: torch.Tensor,
    baseline_vector: torch.Tensor,
    n_permutations: int = 16,
) -> Dict[str, float]:
    """
    Computes Shapley values for 31 players in the DTFD-MIL Fusion framework:
    Player 0 (Image morphology block: 1536-d) and Players 1..30 (individual 30 pathway channels).
    Uses antithetic permutation sampling against an empirical training baseline.

    Args:
        dtfd_fusion_model: Fitted PatientResponseModel (AdaptiveFusion + DTFD_MIL).
        patient_bag: (N, 1566) multimodal patient bag tensor.
        baseline_vector: (1566,) constant feature mean estimated across all training bags.
        n_permutations: Number of permutation samples (antithetic pairs).

    Returns:
        Dictionary mapping player names ('Image', 'Pathway_0', ..., 'Pathway_29')
        to their marginal Shapley attribution values.
    """
    dtfd_fusion_model.eval()

    N = patient_bag.shape[0]
    device = patient_bag.device
    baseline_bag = baseline_vector.to(device).unsqueeze(0).repeat(N, 1)

    players = ["Image"] + [f"Pathway_{k}" for k in range(30)]
    n_players = len(players)
    shapley_values: Dict[str, float] = {p: 0.0 for p in players}

    for _ in range(n_permutations // 2):
        # Draw a random player ordering and its antithetic reverse
        order = np.random.permutation(n_players)
        for perm in [order, order[::-1]]:
            current_bag = baseline_bag.clone()
            prev_prob = dtfd_fusion_model.predict_probability(current_bag)

            for player_idx in perm:
                # Add player feature values from the actual patient bag
                if player_idx == 0:
                    current_bag[:, :1536] = patient_bag[:, :1536]
                else:
                    k = player_idx - 1
                    current_bag[:, 1536 + k] = patient_bag[:, 1536 + k]

                curr_prob = dtfd_fusion_model.predict_probability(current_bag)
                marginal_contribution = curr_prob - prev_prob
                shapley_values[players[player_idx]] += float(marginal_contribution / n_permutations)
                prev_prob = curr_prob

    return shapley_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HistoSPAR Feature Attribution")
    parser.add_argument("--channel", type=int, default=0, help="Target pathway channel (0-29)")
    parser.add_argument("--steps", type=int, default=64, help="Gauss-Legendre integration steps")
    parser.add_argument("--permutations", type=int, default=8, help="Shapley antithetic permutations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HistoSPAR] Initializing Feature Attribution on {device}...")

    encoder = PatchEncoder(arm="Fusion", image_dim=1536, pathway_dim=30, hidden_dim=512).to(device)
    mil_model = DTFD_MIL(feature_dim=512, n_classes=2).to(device)
    response_model = PatientResponseModel(encoder, mil_model)

    # Reference patient bag
    patient_bag = torch.randn(80, 1566, device=device)
    image_features = patient_bag[:, :1536]
    pathway_scores = patient_bag[:, 1536:]

    # 1. Conditional IG
    attr_val, patch_map = conditional_pathway_ig(
        response_model, image_features, pathway_scores, target_channel_k=args.channel, steps=args.steps
    )
    print(f"Conditional IG for Pathway Channel {args.channel}:")
    print(f"  Patient-Level Attribution Score: {attr_val:+.4e}")
    print(f"  Regional Attributions Map Shape: {patch_map.shape}")

    # 2. Grouped Shapley
    baseline_vec = torch.zeros(1566, device=device)
    shapley_res = grouped_shapley_attribution(
        response_model, patient_bag, baseline_vec, n_permutations=args.permutations
    )
    print("\nGrouped Shapley Attribution (Sample Players):")
    print(f"  Image Morphology Block: {shapley_res['Image']:+.4e}")
    for k in range(min(5, 30)):
        print(f"  Pathway_{k}: {shapley_res[f'Pathway_{k}']:+.4e}")


if __name__ == "__main__":
    main()
