"""Patch-SIFT / Overlapping Block Shift-Vector Copy-Move Forgery Detection Engine.

Implements exact + near-exact overlapping block shift-vector matching using:
- Resolution normalization to max 400px on the longest dimension
- 16x16 overlapping block extraction with stride of 4
- Low-variance background block filtering (std < 5.0)
- 8-element normalized feature vectors (zero-mean 4x2 / 2x2 sub-block downsampling)
- Fast nearest-neighbor search via scipy.spatial.cKDTree (excluding self)
- Descriptor distance (< 0.08) and spatial Euclidean distance filtering (> 24px)
- Shift-vector quantization to multiples of 4: (round(abs(x2-x1)/4)*4, round(abs(y2-y1)/4)*4)
- Top shift frequency analysis: flags is_forged = True whenever top shift appears >= 6 times.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .base import BaseForgeryEngine, create_empty_mask, rgb_to_gray


class PatchSIFTEngine(BaseForgeryEngine):
    """Overlapping block shift-vector matching engine for robust copy-move localization."""

    def __init__(
        self,
        block_size: int = 16,
        stride: int = 4,
        max_side: int = 400,
        min_std: float = 5.0,
        dist_threshold: float = 0.08,
        min_spatial_dist: float = 24.0,
        min_shift_count: int = 6,
        **kwargs: Any,
    ):
        super().__init__(name="Patch-SIFT")
        self.block_size = block_size
        self.stride = stride
        self.max_side = max_side
        self.min_std = min_std
        self.dist_threshold = dist_threshold
        self.min_spatial_dist = min_spatial_dist
        self.min_shift_count = min_shift_count

    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        gray = rgb_to_gray(image_rgb)

        # 1. Resize grayscale image to max 400px on the longest side
        max_dim = max(h, w)
        if max_dim > self.max_side:
            scale = self.max_side / float(max_dim)
            proc_w = max(16, int(round(w * scale)))
            proc_h = max(16, int(round(h * scale)))
            resized_gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            proc_w, proc_h = w, h
            resized_gray = gray.copy()

        inv_scale = 1.0 / scale
        orig_bs = max(4, int(round(self.block_size * inv_scale)))

        if proc_w < self.block_size or proc_h < self.block_size:
            return {
                "is_forged": False,
                "confidence": 0.0,
                "visual_result": image_rgb.copy(),
                "intermediate_view": image_rgb.copy(),
                "binary_mask": create_empty_mask(h, w),
                "metrics": {
                    "total_blocks": 0,
                    "matched_pairs": 0,
                    "top_shift_vector": [0, 0],
                    "top_shift_count": 0,
                    "cloned_blocks": 0,
                    "ransac_inliers": 0,
                    "cloned_pairs": 0,
                },
            }

        # 2. Extract 16x16 blocks with stride 4 & filter flat background blocks (std < 5.0)
        blocks: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []
        bs = self.block_size

        for y in range(0, proc_h - bs + 1, self.stride):
            for x in range(0, proc_w - bs + 1, self.stride):
                b = resized_gray[y : y + bs, x : x + bs].astype(np.float32)
                b_std = float(np.std(b))
                if b_std < self.min_std:
                    continue

                # 8-element feature vector: zero-mean 4x2 / 2x2 sub-block downsampling
                zm = b - float(np.mean(b))
                feat = cv2.resize(zm, (4, 2), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)

                norm = float(np.linalg.norm(feat))
                if norm > 1e-7:
                    feat /= norm

                blocks.append(feat)
                coords.append((x, y))

        # Intermediate view: sampled block grid
        intermediate_view = image_rgb.copy()
        step = max(1, len(coords) // 150)
        for idx in range(0, len(coords), step):
            cx, cy = coords[idx]
            ox = int(round(cx * inv_scale))
            oy = int(round(cy * inv_scale))
            cv2.rectangle(
                intermediate_view,
                (ox, oy),
                (min(w - 1, ox + orig_bs), min(h - 1, oy + orig_bs)),
                (60, 200, 140),
                1,
            )

        if len(blocks) < 4:
            return {
                "is_forged": False,
                "confidence": 0.0,
                "visual_result": image_rgb.copy(),
                "intermediate_view": intermediate_view,
                "binary_mask": create_empty_mask(h, w),
                "metrics": {
                    "total_blocks": len(blocks),
                    "matched_pairs": 0,
                    "top_shift_vector": [0, 0],
                    "top_shift_count": 0,
                    "cloned_blocks": 0,
                    "ransac_inliers": 0,
                    "cloned_pairs": 0,
                },
            }

        # 3. cKDTree Nearest-Neighbor search (k=2: index 0 is self, index 1 is nearest neighbor)
        blocks_arr = np.array(blocks, dtype=np.float32)
        tree = cKDTree(blocks_arr)
        distances, indices = tree.query(blocks_arr, k=2, workers=-1)

        # 4. Filter pairs: descriptor distance < threshold AND spatial distance > 24px
        shift_to_pairs: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
        total_filtered_pairs = 0

        for i in range(len(blocks_arr)):
            j = int(indices[i, 1])
            if i < j:  # Process each undirected pair once
                d = float(distances[i, 1])
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                sp_dist = float(np.hypot(x2 - x1, y2 - y1))

                if d < self.dist_threshold and sp_dist > self.min_spatial_dist:
                    # Rounded shift vector: (round(abs(x2-x1)/4)*4, round(abs(y2-y1)/4)*4)
                    sv = (
                        int(round(abs(x2 - x1) / 4.0) * 4),
                        int(round(abs(y2 - y1) / 4.0) * 4),
                    )
                    if sv not in shift_to_pairs:
                        shift_to_pairs[sv] = []
                    shift_to_pairs[sv].append(((x1, y1), (x2, y2)))
                    total_filtered_pairs += 1

        # 5. Top shift vector frequency analysis
        if shift_to_pairs:
            top_shift, top_pairs = max(shift_to_pairs.items(), key=lambda item: len(item[1]))
            top_count = len(top_pairs)
        else:
            top_shift = (0, 0)
            top_pairs = []
            top_count = 0

        # Condition: Top shift vector appears >= min_shift_count (default 6)
        is_forged = (top_count >= self.min_shift_count)

        # 6. Visualization & Localization Mask
        visual_result = image_rgb.copy()
        binary_mask = create_empty_mask(h, w)

        if is_forged:
            confidence = float(min(99.5, 75.0 + top_count * 1.5))

            for (x1, y1), (x2, y2) in top_pairs:
                ox1 = int(round(x1 * inv_scale))
                oy1 = int(round(y1 * inv_scale))
                ox2 = int(round(x2 * inv_scale))
                oy2 = int(round(y2 * inv_scale))

                # Highlight blocks on binary mask
                binary_mask[oy1 : min(h, oy1 + orig_bs), ox1 : min(w, ox1 + orig_bs)] = 1.0
                binary_mask[oy2 : min(h, oy2 + orig_bs), ox2 : min(w, ox2 + orig_bs)] = 1.0

                # Draw green connecting lines between matching blocks
                pt1 = (min(w - 1, ox1 + orig_bs // 2), min(h - 1, oy1 + orig_bs // 2))
                pt2 = (min(w - 1, ox2 + orig_bs // 2), min(h - 1, oy2 + orig_bs // 2))
                cv2.line(visual_result, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(visual_result, pt1, 3, (255, 60, 60), -1)
                cv2.circle(visual_result, pt2, 3, (60, 60, 255), -1)

            # Smooth binary mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            binary_mask = cv2.GaussianBlur(binary_mask, (15, 15), 0)
            binary_mask = np.clip(binary_mask, 0.0, 1.0).astype(np.float32)
        else:
            confidence = float(min(15.0, top_count * 2.0))

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "visual_result": visual_result,
            "intermediate_view": intermediate_view,
            "binary_mask": binary_mask,
            "metrics": {
                "total_blocks": len(blocks),
                "matched_pairs": total_filtered_pairs,
                "top_shift_vector": list(top_shift),
                "top_shift_count": top_count,
                "cloned_blocks": top_count if is_forged else 0,
                "ransac_inliers": 0,
                "cloned_pairs": 0,
            },
        }
