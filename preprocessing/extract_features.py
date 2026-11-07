"""
Feature extraction pipeline using frozen UNI2-h for HistoSPAR.
Extracts 3,072-dimensional joint multiview representations across 8 dihedral views
and single-view 1,536-dimensional CLS tokens.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def dihedral_eight_views(img_tensor: torch.Tensor) -> List[torch.Tensor]:
    """Generates all 8 views under the D4 dihedral group (4 rotations x 2 reflections)."""
    views: List[torch.Tensor] = []
    for flipped in [img_tensor, TF.hflip(img_tensor)]:
        for k in [0, 1, 2, 3]:
            views.append(torch.rot90(flipped, k=k, dims=[-2, -1]))
    return views


def histology_representation(
    regions: Sequence[Dict[str, Any]],
    uni2h: nn.Module,
    readout: str = "joint_multiview",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encodes H&E tissue patches using frozen UNI2-h (ViT-Giant).
    Produces 3072-dimensional joint multiview representations across 8 dihedral views.

    Args:
        regions: Sequence of tissue region dictionaries containing 'patch'.
        uni2h: Pretrained frozen UNI2-h model.
        readout: 'joint_multiview' (champion, 3072-d) or 'single_view_cls' (ablation, 1536-d).

    Returns:
        X: (N, 3072) joint readout for spatial reconstruction.
        G: (N, 1536) single-view CLS for graph similarity.
        F: (N, 1536) single-view CLS for clinical MIL prediction.
    """
    uni2h.eval()
    for param in uni2h.parameters():
        param.requires_grad = False

    readout_features: List[torch.Tensor] = []
    cls_features: List[torch.Tensor] = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for reg in regions:
            patch = torch.from_numpy(reg["patch"]).permute(2, 0, 1).float() / 255.0
            patch = TF.resize(patch, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
            patch = (patch - mean) / std

            # Extract intermediate tokens from layer 24
            cls_0, _ = uni2h.get_intermediate_tokens(patch.unsqueeze(0), layer=24)
            cls_0 = cls_0.squeeze(0)  # (1536,)
            cls_features.append(cls_0)

            if readout == "single_view_cls":
                readout_features.append(cls_0)
            else:
                views: List[torch.Tensor] = []
                for v in dihedral_eight_views(patch):
                    cls_v, patch_tokens_v = uni2h.get_intermediate_tokens(v.unsqueeze(0), layer=24)
                    joint_v = torch.cat([cls_v.squeeze(0), patch_tokens_v.squeeze(0).mean(dim=0)], dim=-1)
                    views.append(joint_v)
                readout_features.append(torch.stack(views, dim=0).mean(dim=0))

    X = torch.stack(readout_features, dim=0)  # (N, 3072)
    G = torch.stack(cls_features, dim=0)      # (N, 1536)
    F = torch.stack(cls_features, dim=0)      # (N, 1536)
    return X, G, F
