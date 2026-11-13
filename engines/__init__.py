"""Multi-Algorithm Image Forgery Detection Engines Package.

Provides a unified interface to classical feature-based, frequency/compression-based,
and deep-learning spatial self-correlation copy-move forgery detection algorithms.
"""

from typing import Dict, Type

from .base import (
    BaseForgeryEngine,
    EngineOutput,
    create_empty_mask,
    ensure_rgb,
    overlay_heatmap,
    rgb_to_gray,
    validate_engine_output,
)
from .dl_selfcorr_engine import DeepLearningSelfCorrEngine
from .ela_engine import ELAEngine
from .orb_engine import ORBEngine
from .patch_sift_engine import PatchSIFTEngine
from .sift_engine import SIFTEngine

# Canonical registry of all available forgery detection engines
AVAILABLE_ENGINES: Dict[str, Type[BaseForgeryEngine]] = {
    "sift": SIFTEngine,
    "orb": ORBEngine,
    "patch_sift": PatchSIFTEngine,
    "ela": ELAEngine,
    "dl_selfcorr": DeepLearningSelfCorrEngine,
}


def get_engine(name: str, **kwargs) -> BaseForgeryEngine:
    """Factory helper to instantiate an engine by registered name."""
    canonical_name = name.strip().lower()
    if canonical_name not in AVAILABLE_ENGINES:
        raise KeyError(
            f"Unknown engine '{name}'. Available engines: {list(AVAILABLE_ENGINES.keys())}"
        )
    return AVAILABLE_ENGINES[canonical_name](**kwargs)


__all__ = [
    "BaseForgeryEngine",
    "EngineOutput",
    "ensure_rgb",
    "rgb_to_gray",
    "create_empty_mask",
    "overlay_heatmap",
    "validate_engine_output",
    "SIFTEngine",
    "ORBEngine",
    "PatchSIFTEngine",
    "ELAEngine",
    "DeepLearningSelfCorrEngine",
    "AVAILABLE_ENGINES",
    "get_engine",
]

