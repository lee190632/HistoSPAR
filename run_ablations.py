"""
Stage 5: Systematic Ablation Study Suite for HistoSPAR.
Evaluates 6 targeted architectural dimensions:
1. Spatial sampling density (dense windows vs. regular/random thinning)
2. Neighborhood residual mechanisms (full C+B vs. distance-only vs. center-only C)
3. Direct 30-pathway prediction vs. two-stage gene aggregation
4. Spatial alignment & train-inference permutation matrix (4-way evaluation on DTFD-MIL)
5. Functional pathway group deletions vs. size-matched random controls on DTFD-MIL

Usage:
    python run_ablations.py
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_clinical import predict_pcr, train_pcr_predictor
from utils.metrics import compute_clinical_metrics


def sample_spatial_density(
    coordinates: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    mode: str = "dense",
    factor: int = 2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ablation 1: Spatial sampling density protocol.
    Compares dense spatial windows vs. regular grid thinning vs. random thinning.

    Args:
        coordinates: (N, 2) physical coordinates.
        features: (N, D) histology representations.
        targets: (N, 30) ground-truth pathway targets.
        mode: 'dense', 'regular_thinning', or 'random_thinning'.
        factor: Thinning factor along coordinate axes for regular thinning.
        seed: Random seed for reproducible stochastic subsampling.

    Returns:
        Subsampled (coordinates, features, targets).
    """
    N = coordinates.shape[0]
    if mode == "dense":
        return coordinates, features, targets
    elif mode == "regular_thinning":
        coords_np = coordinates.cpu().numpy()
        step = 336.0 * factor
        keep = ((coords_np[:, 0] // step) % 2 == 0) & ((coords_np[:, 1] // step) % 2 == 0)
        idx = torch.from_numpy(np.where(keep)[0])
        return coordinates[idx], features[idx], targets[idx]
    elif mode == "random_thinning":
        rng = np.random.RandomState(seed)
        sample_size = N // (factor * factor)
        idx = torch.from_numpy(rng.choice(N, size=sample_size, replace=False))
        return coordinates[idx], features[idx], targets[idx]
    return coordinates, features, targets


def build_distance_only_neighborhood(
    coordinates: torch.Tensor,
    native_step: float = 336.0,
    radius_steps: float = 2.0,
    max_neighbors: int = 12,
    sigma_d: float = 0.5,
    b_0: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ablation 2 helper: Constructs spatial adjacency using Euclidean distance kernel only (no morphology cosine similarity).
    """
    N = coordinates.shape[0]
    neighbor_idx = torch.zeros((N, max_neighbors), dtype=torch.long, device=coordinates.device)
    neighbor_weights = torch.zeros((N, max_neighbors), dtype=torch.float32, device=coordinates.device)

    for i in range(N):
        dist = torch.norm(coordinates - coordinates[i], dim=-1)
        valid = (dist > 0) & (dist <= radius_steps * native_step)
        cand = torch.nonzero(valid).squeeze(-1)

        if cand.numel() == 0:
            continue

        sorted_order = torch.argsort(dist[cand])[:max_neighbors]
        chosen = cand[sorted_order]
        K = chosen.numel()

        rho = dist[chosen] / native_step
        w_dist = torch.exp(-rho**2 / (2.0 * sigma_d**2))
        a_ij = w_dist / (b_0 + w_dist.sum())

        neighbor_idx[i, :K] = chosen
        neighbor_weights[i, :K] = a_ij

    return neighbor_idx, neighbor_weights


class SpatialPathwayAblation(nn.Module):
    """
    Ablation 2: Residual component network supporting 'full' (C+B),
    'distance_only' (distance graph), and 'center_only' (C alone without spatial residual B).
    """
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        output_dim: int = 30,
        mode: str = "full",
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.H = nn.Linear(input_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, output_dim, bias=True)

        if mode in ("full", "distance_only"):
            self.B = nn.Linear(hidden_dim, output_dim, bias=False)
            nn.init.zeros_(self.B.weight)
        else:
            self.B = None

        self.dropout_p = dropout_p

    def forward(
        self,
        X: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
        neighbor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.gelu(self.H(X), approximate="none")
        h_center = F.dropout(h, p=self.dropout_p, training=self.training)
        y_center = self.C(h_center)

        if self.mode == "center_only" or self.B is None or neighbor_idx is None or neighbor_weights is None:
            return y_center

        h_neighbors = h[neighbor_idx]
        delta_h = h_neighbors - h.unsqueeze(1)
        u = (neighbor_weights.unsqueeze(-1) * delta_h).sum(dim=1)
        y_residual = self.B(u)
        return y_center + y_residual


def permute_within_slide(
    patient_bags: Dict[Any, torch.Tensor],
    seed: Optional[int] = None,
) -> Dict[Any, torch.Tensor]:
    """
    Ablation 4 & 5 helper: Permutes pathway score rows relative to morphology rows
    independently within each patient bag, disrupting fine-grained spatial co-localization.
    """
    rng = np.random.RandomState(seed)
    permuted_bags: Dict[Any, torch.Tensor] = {}

    for pid, bag in patient_bags.items():
        N = bag.shape[0]
        perm_idx = rng.permutation(N)
        new_bag = bag.clone()
        new_bag[:, 1536:] = bag[perm_idx, 1536:]
        permuted_bags[pid] = new_bag

    return permuted_bags


def delete_pathway_channels(
    patient_bags: Dict[Any, torch.Tensor],
    channel_indices: Sequence[int],
) -> Dict[Any, torch.Tensor]:
    """
    Ablation 6 helper: Zeroes out specified pathway channels from patient bags.
    """
    modified_bags: Dict[Any, torch.Tensor] = {}

    for pid, bag in patient_bags.items():
        new_bag = bag.clone()
        for k in channel_indices:
            new_bag[:, 1536 + k] = 0.0
        modified_bags[pid] = new_bag

    return modified_bags


def run_permutation_matrix_ablation(
    train_bags: Dict[Any, torch.Tensor],
    test_bags: Dict[Any, torch.Tensor],
    labels: Dict[Any, int],
    train_ids: Sequence[Any],
    val_ids: Sequence[Any],
    test_ids: Sequence[Any],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Ablation 5: Evaluates cross-modal spatial coupling in DTFD-MIL across four
    train/inference permutation states:
        - Matched/Matched
        - Matched/Permuted
        - Permuted/Matched
        - Permuted/Permuted
    """
    matrix_results: Dict[str, Dict[str, float]] = {}

    for train_mode, test_mode in [
        ("matched", "matched"),
        ("matched", "permuted"),
        ("permuted", "matched"),
        ("permuted", "permuted"),
    ]:
        train_input = permute_within_slide(train_bags, seed=seed) if train_mode == "permuted" else train_bags
        test_input = permute_within_slide(test_bags, seed=seed + 1) if test_mode == "permuted" else test_bags

        mil_model, encoder = train_pcr_predictor(
            train_input, labels, train_ids, val_ids, arm="Fusion", max_epochs=20
        )
        test_probs = [predict_pcr(mil_model, encoder, test_input[pid]) for pid in test_ids]
        test_labels = [labels[pid] for pid in test_ids]

        setting_name = f"{train_mode}_train_{test_mode}_test"
        matrix_results[setting_name] = compute_clinical_metrics(test_labels, test_probs)

    return matrix_results


def run_pathway_group_deletion_ablation(
    patient_bags: Dict[Any, torch.Tensor],
    labels: Dict[Any, int],
    pathway_groups: Dict[str, Sequence[int]],
    train_ids: Sequence[Any],
    val_ids: Sequence[Any],
    test_ids: Sequence[Any],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Ablation 6: Deletes functional pathway subsets from patient bags and retrains
    the DTFD-MIL Fusion model from scratch.
    Compares targeted biological deletions against size-matched random deletions.
    """
    deletion_results: Dict[str, Dict[str, float]] = {}
    rng = np.random.RandomState(seed)
    test_labels = [labels[pid] for pid in test_ids]

    for group_name, channel_indices in pathway_groups.items():
        k = len(channel_indices)

        # 1. Targeted biological deletion
        bags_targeted = delete_pathway_channels(patient_bags, channel_indices)
        mil_model_t, encoder_t = train_pcr_predictor(
            bags_targeted, labels, train_ids, val_ids, arm="Fusion", max_epochs=20
        )
        probs_t = [predict_pcr(mil_model_t, encoder_t, bags_targeted[pid]) for pid in test_ids]
        deletion_results[f"{group_name}_deleted"] = compute_clinical_metrics(test_labels, probs_t)

        # 2. Size-matched random deletion control
        random_indices = rng.choice(30, size=k, replace=False).tolist()
        bags_random = delete_pathway_channels(patient_bags, random_indices)
        mil_model_r, encoder_r = train_pcr_predictor(
            bags_random, labels, train_ids, val_ids, arm="Fusion", max_epochs=20
        )
        probs_r = [predict_pcr(mil_model_r, encoder_r, bags_random[pid]) for pid in test_ids]
        deletion_results[f"{group_name}_random_matched"] = compute_clinical_metrics(test_labels, probs_r)

    return deletion_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HistoSPAR Ablation Studies")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HistoSPAR] Initializing Systematic Ablation Suite on {device}...")

    # Reference bags
    patient_bags = {
        f"P_{i}": torch.randn(60, 1566, device=device) for i in range(10)
    }
    labels = {f"P_{i}": int(i % 2 == 0) for i in range(10)}
    train_ids = [f"P_{i}" for i in range(6)]
    val_ids = ["P_6", "P_7"]
    test_ids = ["P_8", "P_9"]

    # 1. Residual component ablation demonstration
    print("\n--- Ablation 2: Residual Component Architectures ---")
    X = torch.randn(20, 3072, device=device)
    for mode in ["full", "distance_only", "center_only"]:
        ablation_model = SpatialPathwayAblation(mode=mode).to(device)
        print(f"  Initialized Spatial Model mode='{mode}', Residual head B is {type(ablation_model.B)}")

    # 2. Permutation matrix demonstration
    print("\n--- Ablation 5: 4-Way Cross-Modal Permutation Matrix (DTFD-MIL) ---")
    matrix_res = run_permutation_matrix_ablation(
        patient_bags, patient_bags, labels, train_ids, val_ids, test_ids, seed=args.seed
    )
    for state, met in matrix_res.items():
        print(f"  Setting {state}: AUROC = {met['AUROC']:.4f}, AP = {met['AP']:.4f}")

    print("\n[HistoSPAR] Ablation Protocol Verification Complete.")


if __name__ == "__main__":
    main()
