"""
Stage 1: Train Spatial Pathway Reconstruction Model for HistoSPAR.
Learns to reconstruct 30 continuous spatial pathway scores from H&E histology
via frozen UNI2-h representations and a dual-readout (C + B) residual network.

Usage:
    python train_spatial.py --max_epochs 120 --lr 1e-4 --batch_size 32
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from models.spatial_reconstruction import (
    SpatialPathwayReconstruction,
    build_neighborhood,
    predict_pathways_from_histology,
    train_pathway_reconstruction,
)
from utils.metrics import compute_spatial_metrics


def run_spatial_training(
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
) -> Tuple[SpatialPathwayReconstruction, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Executes end-to-end training of the spatial pathway reconstruction model
    and evaluates multi-dimensional spatial reconstruction metrics.
    """
    model, mu, sigma = train_pathway_reconstruction(
        features=features,
        raw_targets=raw_targets,
        coordinates=coordinates,
        patient_ids=patient_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        native_step=native_step,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
    )

    # Evaluate validation spatial metrics
    neighbor_idx, neighbor_weights = build_neighborhood(
        coordinates, features[:, :1536], native_step=native_step
    )
    model.eval()
    with torch.no_grad():
        val_preds_z = model(features, neighbor_idx, neighbor_weights)[val_idx]

    metrics = compute_spatial_metrics(
        raw_truth=raw_targets[val_idx].cpu().numpy(),
        pred_z=val_preds_z.cpu().numpy(),
        mu=mu.cpu().numpy(),
        sigma=sigma.cpu().numpy(),
        patient_ids=[patient_ids[int(i)] for i in val_idx],
    )
    return model, mu, sigma, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HistoSPAR Spatial Pathway Reconstruction")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=120, help="Maximum training epochs")
    parser.add_argument("--native_step", type=float, default=336.0, help="Native step width in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HistoSPAR] Initializing Stage 1 Spatial Reconstruction on {device}...")

    # Synthetic reference execution for verification / pipeline check
    N_regions = 100
    features = torch.randn(N_regions, 3072, device=device)
    raw_targets = torch.randn(N_regions, 30, device=device)
    coordinates = torch.rand(N_regions, 2, device=device) * 1000.0
    patient_ids = [f"Patient_{i // 20}" for i in range(N_regions)]

    train_idx = torch.arange(80, device=device)
    val_idx = torch.arange(80, N_regions, device=device)

    model, mu, sigma, metrics = run_spatial_training(
        features=features,
        raw_targets=raw_targets,
        coordinates=coordinates,
        patient_ids=patient_ids,
        train_idx=train_idx,
        val_idx=val_idx,
        native_step=args.native_step,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
    )

    print(f"[HistoSPAR] Stage 1 Training Complete.")
    print(f"Validation Spatial Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
