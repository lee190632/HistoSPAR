"""
Statistical metrics and clinical threshold calibration utilities for HistoSPAR.
Provides pure mathematical and statistical evaluation functions for spatial reconstruction
fidelity and patient-level clinical prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def compute_patient_pathway_macro_pcc(
    truth_z: np.ndarray,
    pred_z: np.ndarray,
    patient_ids: Sequence[Any],
) -> float:
    """
    Computes Patient-Pathway Macro Pearson Correlation Coefficient (PCC).
    Averages spatial correlations within each patient and each pathway independently.

    Args:
        truth_z: (N, K) standardized ground-truth pathway targets.
        pred_z: (N, K) reconstructed standardized pathway scores.
        patient_ids: (N,) patient donor identifier for each region.

    Returns:
        Macro PCC averaged equally across patients.
    """
    unique_patients = np.unique(patient_ids)
    patient_correlations: List[float] = []

    for pid in unique_patients:
        mask = (np.array(patient_ids) == pid)
        if np.sum(mask) < 3:
            continue
        pathway_pccs: List[float] = []
        for k in range(truth_z.shape[1]):
            r, _ = pearsonr(truth_z[mask, k], pred_z[mask, k])
            if not np.isnan(r):
                pathway_pccs.append(float(r))
        if len(pathway_pccs) > 0:
            patient_correlations.append(float(np.mean(pathway_pccs)))

    return float(np.mean(patient_correlations)) if patient_correlations else 0.0


def compute_spatial_metrics(
    raw_truth: np.ndarray,
    pred_z: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    patient_ids: Sequence[Any],
) -> Dict[str, float]:
    """
    Evaluates multi-dimensional spatial pathway reconstruction performance.

    Args:
        raw_truth: (N, 30) ground-truth raw ssGSEA scores.
        pred_z: (N, 30) reconstructed standardized pathway scores.
        mu: (30,) training target means.
        sigma: (30,) training target sample standard deviations.
        patient_ids: (N,) donor patient identifier for each region.

    Returns:
        Dictionary of Macro PCC, Flat PCC, zMSE, and raw MAE.
    """
    truth_z = (raw_truth - mu) / sigma
    pred_raw = pred_z * sigma + mu

    macro_pcc = compute_patient_pathway_macro_pcc(truth_z, pred_z, patient_ids)
    flat_pcc, _ = pearsonr(truth_z.ravel(), pred_z.ravel())
    zmse = mean_squared_error(truth_z, pred_z)
    raw_mae = mean_absolute_error(raw_truth, pred_raw)

    return {
        "Macro_PCC": float(macro_pcc),
        "Flat_PCC": float(flat_pcc),
        "zMSE": float(zmse),
        "raw_MAE": float(raw_mae),
    }


def compute_clinical_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Computes patient-level discrimination and classification metrics at a given decision threshold.

    Args:
        labels: (N,) binary response labels (1 = pCR, 0 = non-pCR).
        probabilities: (N,) predicted pCR probabilities.
        threshold: Binary decision cutoff.

    Returns:
        Dictionary containing AUROC, AP, Accuracy, Sensitivity, Specificity, PPV, NPV, and F1.
    """
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "AUROC": float(auroc),
        "AP": float(ap),
        "Accuracy": float(accuracy),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1),
    }


def select_youden_threshold(
    val_labels: Sequence[int],
    val_probabilities: Sequence[float],
) -> float:
    """
    Calibrates the optimal binary classification threshold by maximizing Youden's Index
    on the internal validation fold:
        J(t) = Sensitivity(t) + Specificity(t) - 1

    Args:
        val_labels: Ground-truth validation response labels.
        val_probabilities: Validation predicted pCR probabilities.

    Returns:
        Optimal threshold t*.
    """
    candidate_thresholds = np.unique(val_probabilities)
    best_j = -1.0
    best_t = 0.50

    for t in candidate_thresholds:
        preds = (np.array(val_probabilities) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t


def summarize_mean_sd(results: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Computes mean and sample standard deviation (ddof=1) across repeated evaluation splits."""
    summary: Dict[str, Dict[str, float]] = {}
    for setting, runs in results.items():
        summary[setting] = {}
        if not runs:
            continue
        for metric in runs[0].keys():
            vals = [r[metric] for r in runs if not np.isnan(r[metric])]
            if vals:
                summary[setting][f"{metric}_mean"] = float(np.mean(vals))
                summary[setting][f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary
