"""Base module for Multi-Algorithm Image Forgery Detection Engines.

Defines the standard output dictionary structure, BaseForgeryEngine abstract class,
and shared computer vision forensic utility functions.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
from PIL import Image


class EngineOutput(TypedDict):
    """Standardized output dictionary for all forgery detection engines."""
    name: str
    is_forged: bool
    confidence: float  # 0.0 to 100.0
    execution_time_ms: float
    visual_result: np.ndarray  # RGB uint8 image (H, W, 3)
    intermediate_view: np.ndarray  # RGB uint8 image (H, W, 3)
    binary_mask: np.ndarray  # float32 2D array (H, W) normalized [0.0, 1.0]
    metrics: Dict[str, Any]  # Algorithm-specific statistics


def ensure_rgb(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Ensures the input image is an RGB uint8 NumPy array of shape (H, W, 3)."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported array dimensions: {image.ndim}")
    else:
        raise TypeError(f"Expected numpy.ndarray or PIL.Image.Image, got {type(image)}")

    return np.ascontiguousarray(image, dtype=np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Converts RGB image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def create_empty_mask(h: int, w: int) -> np.ndarray:
    """Creates a blank 2D float32 mask initialized to 0.0."""
    return np.zeros((h, w), dtype=np.float32)


def overlay_heatmap(
    base_image: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.55
) -> np.ndarray:
    """Overlays a float32 mask [0, 1] onto an RGB image using OpenCV colormap.

    Returns:
        RGB uint8 image with glowing overlay.
    """
    h, w = base_image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    norm_mask = np.clip(mask, 0.0, 1.0)
    mask_u8 = (norm_mask * 255).astype(np.uint8)
    
    # OpenCV colormaps produce BGR images, convert to RGB
    colored_bgr = cv2.applyColorMap(mask_u8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Weight blending by mask intensity so low-confidence areas remain original image
    weight = np.clip(norm_mask[..., np.newaxis] * alpha * 1.5, 0.0, 0.85)
    blended = (1.0 - weight) * base_image.astype(np.float32) + weight * colored_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def validate_engine_output(out: Dict[str, Any], expected_shape: Tuple[int, int]) -> EngineOutput:
    """Validates and enforces strict types and ranges for EngineOutput."""
    h, w = expected_shape
    name = str(out.get("name", "UnknownEngine"))
    is_forged = bool(out.get("is_forged", False))
    confidence = float(np.clip(out.get("confidence", 0.0), 0.0, 100.0))
    execution_time_ms = float(out.get("execution_time_ms", 0.0))

    vis = out.get("visual_result")
    if vis is None or not isinstance(vis, np.ndarray) or vis.shape[:2] != (h, w):
        vis = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        vis = ensure_rgb(vis)

    inter = out.get("intermediate_view")
    if inter is None or not isinstance(inter, np.ndarray) or inter.shape[:2] != (h, w):
        inter = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        inter = ensure_rgb(inter)

    bmask = out.get("binary_mask")
    if bmask is None or not isinstance(bmask, np.ndarray) or bmask.shape[:2] != (h, w):
        bmask = np.zeros((h, w), dtype=np.float32)
    else:
        bmask = np.clip(bmask.astype(np.float32), 0.0, 1.0)
        if bmask.ndim == 3:
            bmask = bmask.squeeze(-1)

    metrics = out.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {"raw_metrics": str(metrics)}

    return {
        "name": name,
        "is_forged": is_forged,
        "confidence": confidence,
        "execution_time_ms": execution_time_ms,
        "visual_result": vis,
        "intermediate_view": inter,
        "binary_mask": bmask,
        "metrics": metrics,
    }


class BaseForgeryEngine(ABC):
    """Abstract Base Class for all Image Forgery Detection Engines."""

    def __init__(self, name: str):
        self.name = name

    def detect(self, image: Union[np.ndarray, Image.Image]) -> EngineOutput:
        """Executes forgery detection on an in-memory image.

        Args:
            image: In-memory RGB NumPy array or PIL Image.

        Returns:
            Standardized EngineOutput dictionary.
        """
        img_rgb = ensure_rgb(image)
        h, w = img_rgb.shape[:2]

        t0 = time.perf_counter()
        try:
            raw_result = self._detect(img_rgb)
        except Exception as e:
            # Robust fallback on unexpected algorithm exceptions
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "name": self.name,
                "is_forged": False,
                "confidence": 0.0,
                "execution_time_ms": float(elapsed),
                "visual_result": img_rgb.copy(),
                "intermediate_view": img_rgb.copy(),
                "binary_mask": create_empty_mask(h, w),
                "metrics": {"error": str(e), "status": "failed"},
            }

        elapsed = (time.perf_counter() - t0) * 1000.0
        raw_result["name"] = self.name
        raw_result["execution_time_ms"] = elapsed
        return validate_engine_output(raw_result, (h, w))

    @abstractmethod
    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Internal detection logic implemented by subclasses."""
        raise NotImplementedError

