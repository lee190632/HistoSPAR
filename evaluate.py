"""
Stage 3: Evaluation and Threshold Calibration Protocol for HistoSPAR.
Evaluates the champion DTFD-MIL Fusion model across repeated internal cross-validation folds
and the independent external validation cohort with prospective Youden Index calibration.

Usage:
    python evaluate.py --folds 5
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from utils.metrics import (
    compute_clinical_metrics,
    select_youden_threshold,
    summarize_mean_sd,
)


def run_evaluation_protocol(
    models: Sequence[Any],
    evaluations: Sequence[Any],
    external_data: Any,
) -> Dict[str, Dict[str, float]]:
    """
    Executes the clinical evaluation protocol across 5 internal cross-validation splits
    and the independent external validation cohort.

    Args:
        models: List of fitted (mil_model, encoder) tuples or model instances with .predict().
        evaluations: List of evaluation fold objects containing val_bags, val_labels, test_bags, test_labels.
        external_data: Independent external cohort container with .bags and .labels.

    Returns:
        Summarized performance across internal test, external fixed (0.50), and external Youden thresholds.
    """
    results: Dict[str, List[Dict[str, float]]] = {
        "internal": [],
        "external_fixed": [],
        "external_youden": [],
    }

    for model, eval_split in zip(models, evaluations):
        # 1. Calibrate decision cutoff on internal validation fold
        val_probs = model.predict(eval_split.val_bags)
        youden_t = select_youden_threshold(eval_split.val_labels, val_probs)

        # 2. Evaluate on internal test set
        test_probs = model.predict(eval_split.test_bags)
        results["internal"].append(compute_clinical_metrics(eval_split.test_labels, test_probs))

        # 3. Prospective evaluation on external cohort with fixed 0.50 and frozen Youden cutoff
        ext_probs = model.predict(external_data.bags)
        results["external_fixed"].append(
            compute_clinical_metrics(external_data.labels, ext_probs, threshold=0.50)
        )
        results["external_youden"].append(
            compute_clinical_metrics(external_data.labels, ext_probs, threshold=youden_t)
        )

    return summarize_mean_sd(results)


class MockModel:
    """Mock predictor for pipeline validation."""
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def predict(self, bags: Sequence[Any]) -> List[float]:
        return self.rng.beta(2, 5, size=len(bags)).tolist()


class MockSplit:
    """Mock cross-validation split container."""
    def __init__(self, n_val: int = 15, n_test: int = 20, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        self.val_bags = [None] * n_val
        self.val_labels = (rng.rand(n_val) > 0.7).astype(int).tolist()
        self.test_bags = [None] * n_test
        self.test_labels = (rng.rand(n_test) > 0.7).astype(int).tolist()


class MockExternal:
    """Mock external cohort container."""
    def __init__(self, n_ext: int = 50, seed: int = 100) -> None:
        rng = np.random.RandomState(seed)
        self.bags = [None] * n_ext
        self.labels = (rng.rand(n_ext) > 0.75).astype(int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HistoSPAR Clinical Performance")
    parser.add_argument("--folds", type=int, default=5, help="Number of repeated evaluation splits")
    args = parser.parse_args()

    print(f"[HistoSPAR] Executing Clinical Evaluation Protocol across {args.folds} folds...")
    models = [MockModel(seed=42 + i) for i in range(args.folds)]
    evaluations = [MockSplit(seed=100 + i) for i in range(args.folds)]
    external_data = MockExternal(seed=999)

    summary = run_evaluation_protocol(models, evaluations, external_data)

    for cohort, metrics in summary.items():
        print(f"\nCohort: {cohort.upper()}")
        for k in ["AUROC_mean", "AUROC_sd", "AP_mean", "AP_sd", "Accuracy_mean", "Accuracy_sd"]:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
