"""
Stage 2: Train Clinical pCR Prediction Model for HistoSPAR.
Trains the multimodal DTFD-MIL Fusion framework using decoupled dual optimizers
and patient-level validation log-loss checkpoint selection.

Usage:
    python train_clinical.py --arm Fusion --max_epochs 200 --lr 1e-4
"""
from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import PatchEncoder
from models.aggregators import AB_MIL, CLAM_SB, DTFD_MIL
from utils.metrics import compute_clinical_metrics


def evaluate_validation_loss(
    mil_model: nn.Module,
    encoder: PatchEncoder,
    patient_bags: Dict[Any, torch.Tensor],
    labels: Dict[Any, int],
    val_ids: Sequence[Any],
    criterion: nn.Module,
) -> float:
    """Computes cross-entropy validation loss on patient-level predictions."""
    mil_model.eval()
    encoder.eval()
    val_loss = 0.0
    device = next(encoder.parameters()).device

    with torch.no_grad():
        for pid in val_ids:
            bag = patient_bags[pid].to(device)
            h = encoder(bag)
            if isinstance(mil_model, DTFD_MIL):
                logits, _ = mil_model(h, num_pseudo_bags=4, is_training=False)
            elif isinstance(mil_model, CLAM_SB):
                logits, _ = mil_model(h, label=None)
            elif isinstance(mil_model, AB_MIL):
                logits, _ = mil_model(h)
            else:
                out = mil_model(h)
                logits = out[0] if isinstance(out, (tuple, list)) else out
            val_loss += criterion(logits, torch.tensor([labels[pid]], device=device)).item()

    return val_loss / len(val_ids)


def train_pcr_predictor(
    patient_bags: Dict[Any, torch.Tensor],
    labels: Dict[Any, int],
    train_ids: Sequence[Any],
    val_ids: Sequence[Any],
    arm: str = "Fusion",
    max_epochs: int = 200,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Tuple[DTFD_MIL, PatchEncoder]:
    """
    Trains the DTFD-MIL clinical response predictor with decoupled Tier 1 and Tier 2 optimizers.
    Selects the best checkpoint via internal validation log-loss.
    """
    device = next(iter(patient_bags.values())).device
    encoder = PatchEncoder(arm=arm, hidden_dim=512).to(device)
    mil_model = DTFD_MIL(feature_dim=512, n_classes=2).to(device)

    # Decoupled parameter groups for feature distillation
    tier1_params = (
        list(mil_model.tier1_attention.parameters())
        + list(mil_model.tier1_classifier.parameters())
        + list(encoder.parameters())
    )
    tier2_params = (
        list(mil_model.tier2_attention.parameters())
        + list(mil_model.tier2_classifier.parameters())
    )

    opt_tier1 = torch.optim.Adam(tier1_params, lr=lr, weight_decay=weight_decay)
    opt_tier2 = torch.optim.Adam(tier2_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(max_epochs):
        mil_model.train()
        encoder.train()

        for pid in train_ids:
            bag = patient_bags[pid].to(device)
            label = torch.tensor([labels[pid]], dtype=torch.long, device=device)

            # Feature projection / fusion
            h = encoder(bag)

            # DTFD-MIL forward pass
            patient_logits, tier1_logits = mil_model(h, num_pseudo_bags=6, is_training=True)

            loss_tier1 = criterion(tier1_logits, label.repeat(tier1_logits.shape[0]))
            loss_tier2 = criterion(patient_logits, label)

            # Optimization step 1: update encoder and Tier 1
            opt_tier1.zero_grad()
            loss_tier1.backward()
            torch.nn.utils.clip_grad_norm_(tier1_params, max_norm=5.0)
            opt_tier1.step()

            # Optimization step 2: update Tier 2
            opt_tier2.zero_grad()
            loss_tier2.backward()
            torch.nn.utils.clip_grad_norm_(tier2_params, max_norm=5.0)
            opt_tier2.step()

        # Validation log-loss evaluation and early stopping
        val_loss = evaluate_validation_loss(mil_model, encoder, patient_bags, labels, val_ids, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "mil": copy.deepcopy(mil_model.state_dict()),
                "encoder": copy.deepcopy(encoder.state_dict()),
            }

    if best_state is not None:
        mil_model.load_state_dict(best_state["mil"])
        encoder.load_state_dict(best_state["encoder"])

    return mil_model, encoder


def predict_pcr(
    mil_model: nn.Module,
    encoder: PatchEncoder,
    patient_bag: torch.Tensor,
) -> float:
    """Inference returns the patient-level probability of achieving pCR."""
    mil_model.eval()
    encoder.eval()
    device = next(encoder.parameters()).device

    with torch.no_grad():
        bag = patient_bag.to(device)
        h = encoder(bag)
        if isinstance(mil_model, DTFD_MIL):
            logits, _ = mil_model(h, num_pseudo_bags=4, is_training=False)
        elif isinstance(mil_model, CLAM_SB):
            logits, _ = mil_model(h, label=None)
        elif isinstance(mil_model, AB_MIL):
            logits, _ = mil_model(h)
        else:
            out = mil_model(h)
            logits = out[0] if isinstance(out, (tuple, list)) else out
        pcr_prob = float(F.softmax(logits, dim=-1)[0, 1].item())

    return pcr_prob


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HistoSPAR Clinical Response Predictor")
    parser.add_argument("--arm", type=str, default="Fusion", choices=["Fusion", "Image", "Pathway"])
    parser.add_argument("--max_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HistoSPAR] Initializing Stage 2 Clinical Predictor ({args.arm} arm) on {device}...")

    # Reference execution bags
    patient_bags = {
        f"P_{i}": torch.randn(np.random.randint(50, 150), 1566, device=device)
        for i in range(12)
    }
    labels = {f"P_{i}": int(i % 3 == 0) for i in range(12)}
    train_ids = [f"P_{i}" for i in range(8)]
    val_ids = [f"P_{i}" for i in range(8, 12)]

    mil_model, encoder = train_pcr_predictor(
        patient_bags=patient_bags,
        labels=labels,
        train_ids=train_ids,
        val_ids=val_ids,
        arm=args.arm,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Test inference predictions
    val_probs = [predict_pcr(mil_model, encoder, patient_bags[pid]) for pid in val_ids]
    val_labels = [labels[pid] for pid in val_ids]
    metrics = compute_clinical_metrics(val_labels, val_probs)

    print(f"[HistoSPAR] Stage 2 Training Complete.")
    print("Validation Discrimination Metrics:")
    for k in ["AUROC", "AP", "Accuracy", "Sensitivity", "Specificity"]:
        print(f"  {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
