"""
Visium HD spatial transcriptomics count aggregation and ssGSEA target computation for HistoSPAR.
Aggregates 2-um capture bins into regional tissue tiles and derives 30 rank-weighted continuous pathway targets.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd


def prepare_pathway_targets(
    counts: np.ndarray,
    positions: np.ndarray,
    genes: Sequence[str],
    image: np.ndarray,
    gene_sets: Dict[str, List[str]],
    region_width: int = 336,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Aggregates 2-um Visium HD counts into regional patches and computes
    30 continuous ssGSEA pathway targets with rank weighting (alpha=0.25).

    Args:
        counts: (N_bins, N_genes) raw count matrix.
        positions: (N_bins, 2) pixel or physical coordinates of capture bins.
        genes: Sequence of measured gene identifiers (e.g., 18,132).
        image: Full-resolution registered H&E image array.
        gene_sets: Dictionary mapping 30 pathway names to member gene lists.
        region_width: Tile side length in pixels (~60 um).

    Returns:
        regions: Filtered tissue regions with coordinates and patch arrays.
        expression: (N_regions, N_genes) log1p normalized expression matrix.
        targets: (N_regions, 30) ssGSEA pathway target enrichment scores.
    """
    import gseapy as gp

    H, W = image.shape[:2]
    regions: List[Dict[str, Any]] = []

    # 1. Tile image and filter background regions
    for y in range(0, H - region_width + 1, region_width):
        for x in range(0, W - region_width + 1, region_width):
            patch = image[y : y + region_width, x : x + region_width]
            gray = 0.2989 * patch[:, :, 0] + 0.5870 * patch[:, :, 1] + 0.1140 * patch[:, :, 2]
            tissue_fraction = np.mean(gray < 230)
            if tissue_fraction >= 0.20:
                regions.append({
                    "patch": patch,
                    "coords": np.array([x + region_width / 2.0, y + region_width / 2.0]),
                    "bbox": (x, y, x + region_width, y + region_width),
                })

    N_regions = len(regions)
    N_genes = len(genes)
    regional_counts = np.zeros((N_regions, N_genes), dtype=np.float32)
    n_bins = np.zeros(N_regions, dtype=np.float32)

    # 2. Bin aggregation within regional boundaries
    for i, reg in enumerate(regions):
        x1, y1, x2, y2 = reg["bbox"]
        in_region = (
            (positions[:, 0] >= x1)
            & (positions[:, 0] < x2)
            & (positions[:, 1] >= y1)
            & (positions[:, 1] < y2)
        )
        count_sum = counts[in_region].sum(axis=0)
        bin_count = np.sum(in_region)
        regional_counts[i] = count_sum
        n_bins[i] = max(float(bin_count), 1.0)

    # 3. Size normalization per captured bin count and log1p scaling
    expression = np.log1p(10000.0 * regional_counts / n_bins[:, None])

    # 4. Rank-weighted ssGSEA over full gene background
    expr_df = pd.DataFrame(expression.T, index=genes)
    ssgsea_res = gp.ssgsea(
        data=expr_df,
        gene_sets=gene_sets,
        sample_norm_method="rank",
        weight=0.25,
        min_size=1,
        score="ES",
        permutation_num=0,
    )
    targets = ssgsea_res.res2d.T.values.astype(np.float32)

    return regions, expression, targets
