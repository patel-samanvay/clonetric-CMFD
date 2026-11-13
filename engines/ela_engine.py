"""Error Level Analysis (ELA) Engine with High-Contrast Edge Suppression.

Performs entirely in-memory JPEG recompression via io.BytesIO at quality=90.
Natural high-contrast image edges naturally exhibit elevated DCT quantization error;
this engine suppresses structural Canny/Sobel edges to focus on regional block
inconsistencies and prevent false alarms on authentic sharp contours.
"""

import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from .base import BaseForgeryEngine, create_empty_mask, overlay_heatmap, rgb_to_gray


class ELAEngine(BaseForgeryEngine):
    """Error Level Analysis engine with structural edge suppression and in-memory recompression."""

    def __init__(
        self,
        quality: int = 90,
        scale: float = 12.0,
        std_multiplier: float = 2.4,
        edge_suppression_factor: float = 0.75,
        min_cluster_pixels: int = 250,
    ):
        super().__init__(name="ELA")
        self.quality = quality
        self.scale = scale
        self.std_multiplier = std_multiplier
        self.edge_suppression_factor = edge_suppression_factor
        self.min_cluster_pixels = min_cluster_pixels

    def _compute_edge_suppression_mask(self, gray: np.ndarray) -> np.ndarray:
        """Computes combined Sobel magnitude and dilated Canny edges normalized [0, 1]."""
        # Canny edges with slight dilation
        canny = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
        canny_dilated = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        # Sobel gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        norm_grad = np.clip(grad_mag / 120.0, 0.0, 1.0)

        # Combined edge mask
        edge_mask = np.maximum(canny_dilated, norm_grad)
        return edge_mask

    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        gray = rgb_to_gray(image_rgb)

        # 1. In-memory JPEG recompression using io.BytesIO (Zero disk writes)
        pil_img = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        resaved_img = Image.open(buf)
        resaved_rgb = np.array(resaved_img.convert("RGB"), dtype=np.float32)

        # 2. Compute absolute difference
        diff = np.abs(image_rgb.astype(np.float32) - resaved_rgb)
        ela_map_raw = np.mean(diff, axis=2)  # Shape (H, W)

        # 3. Filter 3: High-Contrast Edge Suppression
        edge_mask = self._compute_edge_suppression_mask(gray)
        # Dampen sharp structural edges so high-contrast contours don't falsely glow
        damping_weight = 1.0 - (self.edge_suppression_factor * edge_mask)
        dampened_ela_map = ela_map_raw * damping_weight

        # Intermediate view: enhanced edge-suppressed ELA difference image
        diff_dampened = diff * damping_weight[..., np.newaxis]
        intermediate_view = np.clip(diff_dampened * self.scale, 0, 255).astype(np.uint8)

        # 4. Statistical Anomaly Localization
        mean_err = float(np.mean(dampened_ela_map))
        std_err = float(np.std(dampened_ela_map))
        max_err = float(np.max(diff))

        threshold = mean_err + self.std_multiplier * std_err
        raw_anomaly = (dampened_ela_map > threshold).astype(np.uint8)

        # Morphological filtering to eliminate isolated single-pixel salt noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_anomaly = cv2.morphologyEx(raw_anomaly, cv2.MORPH_OPEN, kernel)

        # Connected component analysis for coherent compression anomaly clusters
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_anomaly, connectivity=8)

        binary_mask = create_empty_mask(h, w)
        cluster_found = False
        total_cluster_pixels = 0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # Disregard tiny noise and broad whole-image shifts
            if self.min_cluster_pixels <= area <= int(h * w * 0.40):
                binary_mask[labels == i] = 1.0
                cluster_found = True
                total_cluster_pixels += area

        anomaly_ratio = float(total_cluster_pixels / (h * w))

        # 5. Forgery Decision & Confidence
        if cluster_found and anomaly_ratio >= 0.003:
            is_forged = True
            confidence = float(min(90.0, 50.0 + anomaly_ratio * 350.0))
        else:
            is_forged = False
            confidence = float(min(20.0, anomaly_ratio * 200.0))

        # Smooth binary localization mask
        if cluster_found:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            binary_mask = cv2.dilate(binary_mask, kernel_dilate, iterations=1)
            binary_mask = cv2.GaussianBlur(binary_mask, (19, 19), 0)
            binary_mask = np.clip(binary_mask, 0.0, 1.0).astype(np.float32)

        visual_result = overlay_heatmap(image_rgb, binary_mask, colormap=cv2.COLORMAP_JET, alpha=0.6)

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "visual_result": visual_result,
            "intermediate_view": intermediate_view,
            "binary_mask": binary_mask,
            "metrics": {
                "mean_error": mean_err,
                "max_error": max_err,
                "std_error": std_err,
                "threshold": float(threshold),
                "anomaly_ratio": anomaly_ratio,
                "cluster_pixels": int(total_cluster_pixels),
                "edge_suppression_applied": True,
            },
        }
