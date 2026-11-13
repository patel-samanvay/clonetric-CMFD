"""Real-time Multi-Algorithm Image Forgery Detection Orchestrator.

Manages parallel execution of multiple forensic engines (SIFT, ORB, Patch-SIFT, ELA, DL-SelfCorr)
using ThreadPoolExecutor, applies specificity reliability weights, enforces strict quorum rules
(RANSAC geometric lock >= 8 inliers OR >= 2 distinct engines with > 15% spatial mask overlap),
and renders consensus forensic overlays.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from engines import (
    AVAILABLE_ENGINES,
    BaseForgeryEngine,
    EngineOutput,
    ensure_rgb,
    get_engine,
)

# Filter 4: Reliability Specificity Weights
DEFAULT_ENGINE_WEIGHTS = {
    "sift": 0.35,        # High authority for verified geometric inliers
    "orb": 0.35,         # High authority for verified geometric inliers
    "patch_sift": 0.15,  # Moderate authority for localized patch features
    "dl_selfcorr": 0.10, # Semantic deep correlation
    "ela": 0.05,         # Supporting signal only, never triggers alone
}


def create_glowing_forensic_heatmap(
    base_image: np.ndarray,
    consensus_mask: np.ndarray,
    max_alpha: float = 0.75,
) -> np.ndarray:
    """Overlays a glowing red/yellow forensic heatmap onto an RGB image.

    Args:
        base_image: RGB uint8 NumPy array of shape (H, W, 3).
        consensus_mask: 2D float32 array of shape (H, W) in range [0.0, 1.0].
        max_alpha: Maximum blending alpha for peak confidence regions.

    Returns:
        RGB uint8 NumPy array with glowing forensic visualization.
    """
    h, w = base_image.shape[:2]
    if consensus_mask.shape[:2] != (h, w):
        consensus_mask = cv2.resize(consensus_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    norm_mask = np.clip(consensus_mask.astype(np.float32), 0.0, 1.0)
    if np.max(norm_mask) <= 0.01:
        return base_image.copy()

    # Red-Yellow spectral interpolation: high confidence -> red, moderate -> gold
    glow_rgb = np.zeros((h, w, 3), dtype=np.float32)
    glow_rgb[..., 0] = 255.0  # R
    glow_rgb[..., 1] = 220.0 * np.clip(1.0 - norm_mask * 0.75, 0.1, 1.0)  # G
    glow_rgb[..., 2] = 25.0  # B

    alpha = np.clip(np.power(norm_mask, 0.8) * max_alpha, 0.0, 0.88)[..., np.newaxis]
    blended = (1.0 - alpha) * base_image.astype(np.float32) + alpha * glow_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def calculate_mask_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    threshold: float = 0.15,
) -> Tuple[float, float]:
    """Calculates both IoU and Min-Area-Overlap between two 2D float masks.

    Returns:
        (iou, min_overlap): IoU and overlap relative to the smaller mask.
    """
    bin_a = mask_a > threshold
    bin_b = mask_b > threshold

    inter = float(np.logical_and(bin_a, bin_b).sum())
    union = float(np.logical_or(bin_a, bin_b).sum())
    min_area = float(min(bin_a.sum(), bin_b.sum()))

    iou = (inter / (union + 1e-7)) if union > 0 else 0.0
    min_overlap = (inter / (min_area + 1e-7)) if min_area > 0 else 0.0
    return float(iou), float(min_overlap)


class ForgeryOrchestrator:
    """Orchestrates multi-algorithm real-time forgery detection with smart ensemble gating."""

    def __init__(
        self,
        engines: Optional[List[str]] = None,
        max_dimension: int = 800,
        max_workers: int = 5,
    ):
        self.max_dimension = max_dimension
        self.max_workers = max_workers
        self.engine_keys = engines or list(AVAILABLE_ENGINES.keys())

        # Pre-warm requested engine singletons
        self._engines: Dict[str, BaseForgeryEngine] = {}
        for key in self.engine_keys:
            self._engines[key] = get_engine(key)

    def resize_for_realtime(
        self,
        image: Union[np.ndarray, Image.Image, str],
    ) -> Tuple[np.ndarray, float]:
        """Loads and resizes image cleanly if longest dimension exceeds max_dimension."""
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)

        img_rgb = ensure_rgb(image)
        h, w = img_rgb.shape[:2]
        longest_side = max(h, w)

        if longest_side > self.max_dimension:
            scale = self.max_dimension / float(longest_side)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale

        return img_rgb, 1.0

    def analyze(
        self,
        image: Union[np.ndarray, Image.Image, str],
        selected_engines: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Runs the multi-algorithm detection pipeline and applies smart ensemble gating.

        Returns:
            Dictionary with final verdict, confidence, consensus mask, overlays, and metrics.
        """
        start_time = time.perf_counter()

        # 1. Clean real-time image normalization and resizing
        processed_image, scale_factor = self.resize_for_realtime(image)
        h, w = processed_image.shape[:2]

        active_keys = selected_engines or self.engine_keys

        for key in active_keys:
            if key not in self._engines:
                self._engines[key] = get_engine(key)

        # 2. Parallel multi-engine execution
        results: Dict[str, EngineOutput] = {}
        with ThreadPoolExecutor(max_workers=min(len(active_keys), self.max_workers)) as executor:
            future_to_key = {
                executor.submit(self._engines[k].detect, processed_image): k
                for k in active_keys
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    results[key] = {
                        "name": key,
                        "is_forged": False,
                        "confidence": 0.0,
                        "execution_time_ms": 0.0,
                        "visual_result": processed_image.copy(),
                        "intermediate_view": processed_image.copy(),
                        "binary_mask": np.zeros((h, w), dtype=np.float32),
                        "metrics": {"error": str(e)},
                    }

        # 3. Filter 4: Reliability Specificity Weights for Consensus Mask
        weights = {}
        for k in active_keys:
            if custom_weights and k in custom_weights:
                weights[k] = max(0.01, float(custom_weights[k]))
            else:
                weights[k] = DEFAULT_ENGINE_WEIGHTS.get(k, 0.20)

        weight_sum = sum(weights.values())
        raw_consensus_mask = np.zeros((h, w), dtype=np.float32)
        for k, res in results.items():
            norm_w = weights[k] / weight_sum
            raw_consensus_mask += res["binary_mask"] * norm_w

        raw_consensus_mask = np.clip(raw_consensus_mask, 0.0, 1.0).astype(np.float32)

        # 4. Quorum Rule Evaluation
        # (a) Check for confirmed RANSAC geometric lock (inliers >= 8) on any keypoint engine
        has_ransac_lock = False
        ransac_lock_engine = None
        max_inliers = 0

        for k in ["sift", "orb", "patch_sift"]:
            if k in results and results[k]["is_forged"]:
                m = results[k]["metrics"]
                inliers = int(m.get("ransac_inliers", m.get("cloned_pairs", 0)))
                if inliers > max_inliers:
                    max_inliers = inliers
                if inliers >= 8:
                    has_ransac_lock = True
                    ransac_lock_engine = k

        # (b) Check for multi-engine spatial overlap (> 15% IoU or relative overlap in the same region)
        forged_engines = [k for k, r in results.items() if r["is_forged"]]
        iou_scores = []
        max_overlap_score = 0.0
        has_multi_engine_overlap = False
        overlapping_pairs = []

        if len(forged_engines) >= 2:
            for eng_a, eng_b in itertools.combinations(forged_engines, 2):
                iou, min_ov = calculate_mask_overlap(
                    results[eng_a]["binary_mask"],
                    results[eng_b]["binary_mask"],
                    threshold=0.15,
                )
                iou_scores.append(iou)
                top_ov = max(iou, min_ov)
                if top_ov > max_overlap_score:
                    max_overlap_score = top_ov
                # Quorum requirement: IoU / pixel overlap > 15% (0.15)
                if top_ov > 0.15:
                    has_multi_engine_overlap = True
                    overlapping_pairs.append((eng_a, eng_b, top_ov))

            mean_spatial_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
        else:
            mean_spatial_iou = 0.0

        # Filter 4 Quorum Decision:
        # ONLY output "TAMPERED" if:
        # (a) Confirmed RANSAC geometric lock (inliers >= 8), OR
        # (b) At least 2 distinct engines have overlapping spatial masks (> 15%)
        quorum_passed = has_ransac_lock or has_multi_engine_overlap

        total_active = len(active_keys)
        weighted_conf = sum(results[k]["confidence"] * (weights[k] / weight_sum) for k in active_keys)

        if quorum_passed:
            verdict = "TAMPERED"
            is_forged = True
            consensus_mask = raw_consensus_mask

            if has_ransac_lock and has_multi_engine_overlap:
                confidence = min(99.5, 82.0 + max_inliers * 1.5 + max_overlap_score * 15.0)
            elif has_ransac_lock:
                confidence = min(98.0, 78.0 + max_inliers * 2.0)
            else:
                confidence = min(95.0, 68.0 + max_overlap_score * 25.0)

            heatmap_overlay = create_glowing_forensic_heatmap(processed_image, consensus_mask)
        else:
            verdict = "AUTHENTIC"
            is_forged = False
            # Suppress false alarms on authentic images: no glow, zero consensus mask
            consensus_mask = np.zeros((h, w), dtype=np.float32)
            heatmap_overlay = processed_image.copy()
            confidence = float(min(99.0, max(75.0, 100.0 - weighted_conf)))

        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "is_forged": is_forged,
            "consensus_mask": consensus_mask,
            "heatmap_overlay": heatmap_overlay,
            "spatial_iou": round(mean_spatial_iou, 4),
            "execution_time_ms": round(total_elapsed_ms, 2),
            "scale_factor": round(scale_factor, 4),
            "engine_results": results,
            "metrics": {
                "total_engines": total_active,
                "forged_engines_count": len(forged_engines),
                "forged_engines": forged_engines,
                "quorum_passed": quorum_passed,
                "has_ransac_lock": has_ransac_lock,
                "ransac_lock_engine": ransac_lock_engine,
                "max_inliers": max_inliers,
                "has_multi_engine_overlap": has_multi_engine_overlap,
                "max_overlap_score": round(max_overlap_score, 4),
                "overlapping_pairs": overlapping_pairs,
                "pairwise_ious": [round(x, 4) for x in iou_scores],
            },
        }
