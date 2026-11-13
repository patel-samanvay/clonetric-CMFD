"""SIFT-based Copy-Move Forgery Detection Engine with Shift-Vector Matching.

Uses dense high-sensitivity SIFT keypoint extraction (nfeatures=6000, contrastThreshold=0.01)
with 2NN matching, spatial separation filtering, and rounded shift-vector (dx, dy) grouping.
Mathematically guarantees detection whenever >= 3 pairs share the same shift vector.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base import BaseForgeryEngine, create_empty_mask, rgb_to_gray


class SIFTEngine(BaseForgeryEngine):
    """High-sensitivity SIFT detection engine with shift-vector consensus grouping."""

    def __init__(
        self,
        nfeatures: int = 6000,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.01,
        edge_threshold: float = 10.0,
        sigma: float = 1.5,
        ratio_threshold: float = 0.82,
        min_spatial_dist: float = 20.0,
        shift_bin_size: float = 16.0,
        min_shift_count: int = 3,
        **kwargs: Any,
    ):
        super().__init__(name="SIFT")
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )
        self.ratio_threshold = ratio_threshold
        self.min_spatial_dist = min_spatial_dist
        self.shift_bin_size = shift_bin_size
        self.min_shift_count = min_shift_count

    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        gray = rgb_to_gray(image_rgb)

        # 1. Feature Extraction (nfeatures=6000, contrastThreshold=0.01)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        intermediate_view = cv2.drawKeypoints(
            image_rgb.copy(),
            keypoints or [],
            None,
            color=(255, 180, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        if descriptors is None or len(keypoints) < 4:
            return {
                "is_forged": False,
                "confidence": 0.0,
                "visual_result": image_rgb.copy(),
                "intermediate_view": intermediate_view,
                "binary_mask": create_empty_mask(h, w),
                "metrics": {
                    "total_keypoints": len(keypoints) if keypoints else 0,
                    "matched_pairs": 0,
                    "ransac_inliers": 0,
                    "cloned_pairs": 0,
                    "clusters_found": 0,
                    "top_shift_count": 0,
                    "top_shift_vector": [0, 0],
                },
            }

        # 2. Self-matching with BFMatcher (k=3)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

        raw_pairs: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for m_list in matches:
            # Skip m_list[0] which is self-match (queryIdx == trainIdx)
            ext_matches = [m for m in m_list if m.queryIdx != m.trainIdx]
            if len(ext_matches) >= 2:
                m1, m2 = ext_matches[0], ext_matches[1]
                # Compare m_list[1] and m_list[2] using Lowe's ratio test
                if m1.distance <= self.ratio_threshold * m2.distance:
                    if m1.queryIdx < m1.trainIdx:  # Avoid reciprocal duplicate edges
                        p1 = np.array(keypoints[m1.queryIdx].pt, dtype=np.float32)
                        p2 = np.array(keypoints[m1.trainIdx].pt, dtype=np.float32)
                        dist_spatial = float(np.linalg.norm(p2 - p1))
                        if dist_spatial >= self.min_spatial_dist:
                            raw_pairs.append((p1, p2, float(m1.distance)))

        # 3. Group matched pairs by rounded shift vector (dx, dy)
        shift_groups: Dict[Tuple[int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}
        for p1, p2, _ in raw_pairs:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Canonical direction orientation to treat (p1, p2) and (p2, p1) consistently
            if dx < 0 or (abs(dx) < 1e-4 and dy < 0):
                dx, dy = -dx, -dy

            sv = (
                int(round(dx / self.shift_bin_size) * self.shift_bin_size),
                int(round(dy / self.shift_bin_size) * self.shift_bin_size),
            )
            if sv not in shift_groups:
                shift_groups[sv] = []
            shift_groups[sv].append((p1, p2))

        # 4. Flag is_forged = True whenever >= 3 pairs share the same shift vector
        cloned_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        clusters_found = 0
        max_shift_count = 0
        top_shift_vector = (0, 0)

        for sv, p_list in shift_groups.items():
            cnt = len(p_list)
            if cnt > max_shift_count:
                max_shift_count = cnt
                top_shift_vector = sv
            if cnt >= self.min_shift_count:  # >= 3 pairs
                clusters_found += 1
                cloned_pairs.extend(p_list)

        is_forged = (max_shift_count >= self.min_shift_count)

        # 5. Visualization & Localization Mask
        visual_result = image_rgb.copy()
        binary_mask = create_empty_mask(h, w)

        if is_forged and cloned_pairs:
            confidence = float(min(99.0, 60.0 + len(cloned_pairs) * 3.5))
            src_pts = []
            dst_pts = []

            for p1, p2 in cloned_pairs:
                pt1 = (int(round(p1[0])), int(round(p1[1])))
                pt2 = (int(round(p2[0])), int(round(p2[1])))
                src_pts.append(pt1)
                dst_pts.append(pt2)

                # Draw green line between matched cloned points
                cv2.line(visual_result, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(visual_result, pt1, 4, (255, 50, 50), -1)
                cv2.circle(visual_result, pt2, 4, (50, 50, 255), -1)

                # Paint into localization mask
                cv2.circle(binary_mask, pt1, 20, 1.0, -1)
                cv2.circle(binary_mask, pt2, 20, 1.0, -1)

            # Convex hull around clusters of at least 4 inliers
            for pt_list in (src_pts, dst_pts):
                if len(pt_list) >= 4:
                    pts_arr = np.array(pt_list, dtype=np.int32).reshape((-1, 1, 2))
                    hull = cv2.convexHull(pts_arr)
                    cv2.fillPoly(binary_mask, [hull], 0.85)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            binary_mask = cv2.GaussianBlur(binary_mask, (21, 21), 0)
            binary_mask = np.clip(binary_mask, 0.0, 1.0).astype(np.float32)
        else:
            confidence = float(min(15.0, max_shift_count * 4.0))

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "visual_result": visual_result,
            "intermediate_view": intermediate_view,
            "binary_mask": binary_mask,
            "metrics": {
                "total_keypoints": len(keypoints),
                "matched_pairs": len(raw_pairs),
                "ransac_inliers": len(cloned_pairs),
                "cloned_pairs": len(cloned_pairs),
                "clusters_found": clusters_found,
                "top_shift_count": max_shift_count,
                "top_shift_vector": list(top_shift_vector),
            },
        }
