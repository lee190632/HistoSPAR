"""
Multiple Instance Learning (MIL) aggregators for HistoSPAR.
"""
from __future__ import annotations

from .dtfd_mil import DTFD_MIL, GatedAttention
from .ab_mil import AB_MIL
from .clam_sb import CLAM_SB

__all__ = [
    "GatedAttention",
    "DTFD_MIL",
    "AB_MIL",
    "CLAM_SB",
]
