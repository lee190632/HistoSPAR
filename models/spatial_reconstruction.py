"""
Spatial pathway reconstruction neural architecture and neighborhood graph builder for HistoSPAR.
Couples a center morphology readout C with an unbiased zero-initialized spatial residual head B.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import compute_patient_pathway_macro_pcc
from preprocessing.extract_features import histology_representation


def build_neighborhood(
    coordinates: torch.Tensor,
    cls_features: torch.Tensor,
    native_step: float = 336.0,
    radius_steps: float = 2.0,
    max_neighbors: int = 12,
    sigma_d: float = 0.5,
    tau: float = 0.772,
    b_0: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs adaptive spatial adjacency with distance- and morphology-weighted edge weights.

    Args:
        coordinates: (N, 2) spatial coordinates.
        cls_features: (N, 1536) single-view CLS features.
        native_step: Physical sampling step.
        radius_steps: Multiplier defining neighbor search radius.
        max_neighbors: Maximum neighbor capacity K.
        sigma_d: Gaussian distance kernel scale.
        tau: Morphology cosine temperature.
        b_0: Center self-weight.

    Returns:
        neighbor_idx: (N, K) neighbor index matrix on the same device as coordinates.
        neighbor_weights: (N, K) normalized weight matrix a_ij on the same device as coordinates.
    """
    N = coordinates.shape[0]
    G_norm = F.normalize(cls_features, p=2, dim=-1)

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

        cos_sim = (G_norm[i] * G_norm[chosen]).sum(dim=-1)
        w_morph = torch.exp((cos_sim - 1.0) / tau)

        b_ij = w_dist * w_morph
        a_ij = b_ij / (b_0 + b_ij.sum())

        neighbor_idx[i, :K] = chosen
        neighbor_weights[i, :K] = a_ij

    return neighbor_idx, neighbor_weights


class SpatialPathwayReconstruction(nn.Module):
    """
    Dual-readout neural architecture:
    H: Shared nonlinear projection mapping frozen features to latent space.
    C: Center readout predicting pathway targets from regional morphology.
    B: Zero-initialized spatial head predicting corrections from neighborhood differences (no bias).
    """
    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 256,
        output_dim: int = 30,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.H = nn.Linear(input_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, output_dim, bias=True)
        self.B = nn.Linear(hidden_dim, output_dim, bias=False)

        nn.init.zeros_(self.B.weight)
        self.dropout_p = dropout_p

    def forward(
        self,
        X: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass via vectorized tensor broadcasting:
            X: (N, input_dim) histology representations
            neighbor_idx: (N, K) neighbor index matrix
            neighbor_weights: (N, K) normalized spatial weights a_ij
        """
        h = F.gelu(self.H(X), approximate="none")  # (N, hidden_dim)

        # Center morphology readout
        h_center = F.dropout(h, p=self.dropout_p, training=self.training)
        y_center = self.C(h_center)  # (N, output_dim)

        # Vectorized spatial difference aggregation: u_i = sum_j a_ij * (h_j - h_i)
        h_neighbors = h[neighbor_idx]
        delta_h = h_neighbors - h.unsqueeze(1)  # (N, K, hidden_dim)
        u = (neighbor_weights.unsqueeze(-1) * delta_h).sum(dim=1)  # (N, hidden_dim)

        # Unbiased spatial residual projection
        y_residual = self.B(u)  # (N, output_dim)
        return y_center + y_residual


def train_pathway_reconstruction(
    features: torch.Tensor,
    raw_targets: torch.Tensor,
    coordinates: torch.Tensor,
    patient_ids: Sequence[Any],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    native_step: float = 336.0,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_epochs: int = 120,
) -> Tuple[SpatialPathwayReconstruction, torch.Tensor, torch.Tensor]:
    """
    Trains the SpatialPathwayReconstruction network with MSE on standardized targets.
    Selects optimal weights via validation Patient-Pathway Macro PCC.
    """
    mu = raw_targets[train_idx].mean(dim=0, keepdim=True)
    sigma = raw_targets[train_idx].std(dim=0, unbiased=True, keepdim=True)
    targets_z = (raw_targets - mu) / sigma

    neighbor_idx, neighbor_weights = build_neighborhood(
        coordinates, features[:, :1536], native_step=native_step
    )

    device = features.device
    model = SpatialPathwayReconstruction(
        input_dim=features.shape[1], hidden_dim=256, output_dim=30
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(train_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_macro_pcc = -float("inf")
    best_model_state: Optional[Dict[str, Any]] = None

    for epoch in range(max_epochs):
        model.train()
        for (batch_indices,) in train_loader:
            optimizer.zero_grad()
            preds = model(features, neighbor_idx, neighbor_weights)[batch_indices]
            loss = criterion(preds, targets_z[batch_indices])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(features, neighbor_idx, neighbor_weights)[val_idx]
            val_pcc = compute_patient_pathway_macro_pcc(
                targets_z[val_idx].cpu().numpy(),
                val_preds.cpu().numpy(),
                [patient_ids[int(i)] for i in val_idx],
            )

        if val_pcc > best_macro_pcc:
            best_macro_pcc = val_pcc
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, mu.squeeze(0), sigma.squeeze(0)


def predict_pathways_from_histology(
    regions: Sequence[Dict[str, Any]],
    encoder: nn.Module,
    reconstruction_model: SpatialPathwayReconstruction,
    native_step: float = 336.0,
) -> torch.Tensor:
    """
    Applies the trained spatial pathway reconstruction model to clinical H&E tiles.
    Produces an aligned patient bag pairing morphology and predicted pathway scores (N x 1566).
    """
    X, G, F_img = histology_representation(regions, encoder, readout="joint_multiview")
    coordinates = torch.tensor([r["coords"] for r in regions], dtype=torch.float32, device=X.device)

    neighbor_idx, neighbor_weights = build_neighborhood(
        coordinates, G, native_step=native_step
    )

    reconstruction_model.eval()
    with torch.no_grad():
        Q_path = reconstruction_model(X, neighbor_idx, neighbor_weights)

    return torch.cat([F_img, Q_path], dim=-1)  # (N, 1566)
