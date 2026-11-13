"""ORB-based Copy-Move Forgery Detection Engine with Spatial & RANSAC Filtering.

Extracts Oriented FAST and Rotated BRIEF (ORB) keypoints, performs binary descriptor
matching with Lowe's 2NN ratio test (<= 0.75), spatial distance filtering (>= 5% image diagonal),
shift-vector grouping, and strict RANSAC affine geometric verification (reprojection threshold 5.0).
Only verified RANSAC inliers are drawn and rendered into the binary localization mask.
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base import BaseForgeryEngine, create_empty_mask, rgb_to_gray


class ORBEngine(BaseForgeryEngine):
    """ORB-based high-speed feature matching engine with geometric RANSAC inlier filtering."""

    def __init__(
        self,
        nfeatures: int = 5000,
        scale_factor: float = 1.2,
        nlevels: int = 8,
        edge_threshold: int = 15,
        fast_threshold: int = 12,
        dis_threshold: int = 55,
        spatial_dis_threshold: float = 20.0,
        ratio_threshold: float = 0.75,
        min_inliers: int = 6,
        ransac_threshold: float = 5.0,
        shift_bin_size: float = 28.0,
    ):
        super().__init__(name="ORB")
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=nlevels,
            edgeThreshold=edge_threshold,
            fastThreshold=fast_threshold,
        )
        self.dis_threshold = dis_threshold
        self.spatial_dis_threshold = spatial_dis_threshold
        self.ratio_threshold = ratio_threshold
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.shift_bin_size = shift_bin_size

    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]
        gray = rgb_to_gray(image_rgb)
        diag = float(np.hypot(h, w))
        # Discard any matches whose Euclidean distance is less than 5% of the image diagonal
        min_spatial_dist = max(self.spatial_dis_threshold, 0.05 * diag)

        # 1. Feature Extraction (nfeatures=5000)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        intermediate_view = cv2.drawKeypoints(
            image_rgb.copy(),
            keypoints or [],
            None,
            color=(0, 200, 255),
            flags=0,
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
                },
            }

        # 2. Binary Feature Matching with strict 2NN Ratio Test (<= 0.75) and self-match elimination
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

        raw_pairs: List[Tuple[np.ndarray, np.ndarray, float]] = []
        for m in matches:
            # Exclude self-matches
            ext_matches = [match for match in m if match.queryIdx != match.trainIdx]
            if len(ext_matches) >= 2:
                m1, m2 = ext_matches[0], ext_matches[1]
                if m1.distance <= self.dis_threshold:
                    # Lowe's ratio test on Hamming distance
                    if (m1.distance / (m2.distance + 1e-6)) <= self.ratio_threshold:
                        if m1.queryIdx < m1.trainIdx:
                            p1 = np.array(keypoints[m1.queryIdx].pt, dtype=np.float32)
                            p2 = np.array(keypoints[m1.trainIdx].pt, dtype=np.float32)
                            dist_spatial = float(np.linalg.norm(p1 - p2))
                            # Spatial distance filter (>= 5% image diagonal)
                            if dist_spatial >= min_spatial_dist:
                                raw_pairs.append((p1, p2, float(m1.distance)))

        if len(raw_pairs) < self.min_inliers:
            return {
                "is_forged": False,
                "confidence": min(10.0, len(raw_pairs) * 1.5),
                "visual_result": image_rgb.copy(),
                "intermediate_view": intermediate_view,
                "binary_mask": create_empty_mask(h, w),
                "metrics": {
                    "total_keypoints": len(keypoints),
                    "matched_pairs": len(raw_pairs),
                    "ransac_inliers": 0,
                    "cloned_pairs": 0,
                    "clusters_found": 0,
                },
            }

        # 3. Geometric Verification: Grouping & RANSAC Affine Verification
        cloned_pairs, clusters_found = self._filter_ransac_inliers(raw_pairs)
        inlier_count = len(cloned_pairs)
        is_forged = inlier_count >= self.min_inliers

        if is_forged:
            confidence = min(99.0, 52.0 + inlier_count * 3.5)
        else:
            confidence = min(15.0, inlier_count * 2.0)

        # 4. ONLY draw lines and generate binary_mask for genuine RANSAC inliers!
        visual_result = image_rgb.copy()
        binary_mask = create_empty_mask(h, w)

        if is_forged and cloned_pairs:
            src_pts = []
            dst_pts = []
            for p1, p2 in cloned_pairs:
                pt1 = (int(round(p1[0])), int(round(p1[1])))
                pt2 = (int(round(p2[0])), int(round(p2[1])))
                src_pts.append(pt1)
                dst_pts.append(pt2)

                cv2.line(visual_result, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(visual_result, pt1, 4, (255, 60, 60), -1)
                cv2.circle(visual_result, pt2, 4, (60, 60, 255), -1)

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

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "visual_result": visual_result,
            "intermediate_view": intermediate_view,
            "binary_mask": binary_mask,
            "metrics": {
                "total_keypoints": len(keypoints),
                "matched_pairs": len(raw_pairs),
                "ransac_inliers": inlier_count,
                "cloned_pairs": inlier_count,
                "clusters_found": clusters_found,
            },
        }

    def _filter_ransac_inliers(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray, float]],
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
        """Groups candidate matches and isolates verified RANSAC inliers."""
        if len(pairs) < self.min_inliers:
            return [], 0

        shifts = [p[1] - p[0] for p in pairs]
        bins = [tuple(np.round(s / self.shift_bin_size).astype(int)) for s in shifts]
        counts = Counter(bins)

        verified_pairs_dict = {}
        clusters_found = 0

        # Step A: Shift-vector binning
        for bin_key, count in counts.items():
            if count >= 4:
                cluster_candidates = [
                    (pairs[i][0], pairs[i][1])
                    for i, b in enumerate(bins)
                    if b == bin_key
                ]
                src = np.array([p[0] for p in cluster_candidates], dtype=np.float32)
                dst = np.array([p[1] for p in cluster_candidates], dtype=np.float32)

                try:
                    _, inliers = cv2.estimateAffinePartial2D(
                        src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold
                    )
                    if inliers is not None:
                        inlier_mask = inliers.flatten().astype(bool)
                        inlier_count = int(np.sum(inlier_mask))
                        if inlier_count >= 4:
                            clusters_found += 1
                            for idx, is_inl in enumerate(inlier_mask):
                                if is_inl:
                                    pair_key = (
                                        float(cluster_candidates[idx][0][0]),
                                        float(cluster_candidates[idx][0][1]),
                                        float(cluster_candidates[idx][1][0]),
                                        float(cluster_candidates[idx][1][1]),
                                    )
                                    verified_pairs_dict[pair_key] = cluster_candidates[idx]
                except Exception:
                    pass

        # Step B: Global RANSAC on all candidates if binning didn't reach min_inliers
        if len(verified_pairs_dict) < self.min_inliers and len(pairs) >= self.min_inliers:
            all_src = np.array([p[0] for p in pairs], dtype=np.float32)
            all_dst = np.array([p[1] for p in pairs], dtype=np.float32)
            try:
                _, inliers = cv2.estimateAffinePartial2D(
                    all_src, all_dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold, maxIters=2000
                )
                if inliers is not None:
                    inlier_mask = inliers.flatten().astype(bool)
                    if int(np.sum(inlier_mask)) >= self.min_inliers:
                        clusters_found += 1
                        for idx, is_inl in enumerate(inlier_mask):
                            if is_inl:
                                pair_key = (
                                    float(pairs[idx][0][0]),
                                    float(pairs[idx][0][1]),
                                    float(pairs[idx][1][0]),
                                    float(pairs[idx][1][1]),
                                )
                                verified_pairs_dict[pair_key] = (pairs[idx][0], pairs[idx][1])
            except Exception:
                pass

        return list(verified_pairs_dict.values()), clusters_found
