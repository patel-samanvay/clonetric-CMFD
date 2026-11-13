"""Zero-Shot Deep Learning Spatial Self-Correlation Engine with Variance & Z-Score Filtering.

Uses torchvision.models.mobilenet_v3_small to extract intermediate deep feature maps,
masks out flat/uniform background patches (local std < 10.0), enforces a spatial exclusion
radius (>= 15% of feature map dimensions), and applies statistical Z-score thresholding
(correlation > mean + 2.8 * std) to reliably detect copy-move regions.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .base import BaseForgeryEngine, create_empty_mask, overlay_heatmap, rgb_to_gray

# Global singleton cache for MobileNetV3 feature extractors across devices & cutoffs
_MODEL_CACHE: Dict[Tuple[str, int], torch.nn.Module] = {}


def get_cached_feature_extractor(
    device: torch.device,
    feature_layers_cutoff: int = 3,
) -> torch.nn.Module:
    """Singleton model loader and cache for MobileNetV3 intermediate feature extractor.

    Caches the pre-trained weights in RAM/VRAM so subsequent calls and orchestrator
    invocations never reload weights into memory.
    """
    key = (str(device), feature_layers_cutoff)
    if key not in _MODEL_CACHE:
        full_model = torchvision.models.mobilenet_v3_small(weights="DEFAULT").eval().to(device)
        extractor = torch.nn.Sequential(*list(full_model.features[:feature_layers_cutoff]))
        extractor.eval()
        for p in extractor.parameters():
            p.requires_grad = False
        _MODEL_CACHE[key] = extractor
    return _MODEL_CACHE[key]


class DeepLearningSelfCorrEngine(BaseForgeryEngine):
    """Zero-shot spatial self-correlation engine with variance gating and Z-score thresholding."""

    def __init__(
        self,
        feature_layers_cutoff: int = 3,
        exclusion_fraction: float = 0.15,
        min_variance_std: float = 10.0,
        z_score_multiplier: float = 2.8,
        min_cluster_cells: int = 40,
        device: str = "auto",
    ):
        super().__init__(name="DL-SelfCorr")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.exclusion_fraction = exclusion_fraction
        self.min_variance_std = min_variance_std
        self.z_score_multiplier = z_score_multiplier
        self.min_cluster_cells = min_cluster_cells

        # Load or retrieve cached MobileNetV3-Small (weights < 10MB)
        self.feature_extractor = get_cached_feature_extractor(self.device, feature_layers_cutoff)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _compute_local_variance_mask(self, image_rgb: np.ndarray, hf: int, wf: int) -> np.ndarray:
        """Computes local pixel standard deviation and masks flat patches (std < 10.0)."""
        gray = rgb_to_gray(image_rgb).astype(np.float32)
        # 16x16 window corresponds to ~2x receptive field stride
        ksize = (16, 16)
        mean_img = cv2.boxFilter(gray, -1, ksize)
        sq_mean_img = cv2.boxFilter(gray * gray, -1, ksize)
        local_var = np.maximum(0.0, sq_mean_img - (mean_img * mean_img))
        local_std = np.sqrt(local_var)
        std_grid = cv2.resize(local_std, (wf, hf), interpolation=cv2.INTER_AREA)
        return std_grid >= self.min_variance_std

    def _detect(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h, w = image_rgb.shape[:2]

        # 1. Normalize image tensor
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        normalized_img = (img_tensor - self.mean) / self.std

        # 2. Extract intermediate deep feature map
        with torch.no_grad():
            feat = self.feature_extractor(normalized_img)  # Shape (1, C, Hf, Wf)
            _, c, hf, wf = feat.shape

            # Compute local variance mask for the feature map cells
            texture_mask_2d = self._compute_local_variance_mask(image_rgb, hf, wf)
            texture_mask_flat = torch.from_numpy(texture_mask_2d.flatten()).to(self.device)

            # L2 normalize feature vectors along channel dimension
            feat_norm = F.normalize(feat, p=2, dim=1)
            feat_flat = feat_norm.squeeze(0).view(c, -1).t()  # Shape (N, C) where N = Hf * Wf

            # 3. Spatial Self-Correlation (Cosine Similarity Matrix)
            sim_matrix = torch.mm(feat_flat, feat_flat.t())  # Shape (N, N)

            # Build spatial distance coordinate grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(hf, device=self.device),
                torch.arange(wf, device=self.device),
                indexing="ij",
            )
            coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float()
            spatial_dist = torch.cdist(coords, coords)

            # Filter 2: Spatial neighborhood exclusion radius (at least 15% of feature map dimensions)
            exclusion_radius = max(float(self.exclusion_fraction * max(hf, wf)), 6.0)
            mask_neighborhood = spatial_dist <= exclusion_radius

            sim_matrix_masked = sim_matrix.clone()
            sim_matrix_masked[mask_neighborhood] = -1.0

            # Filter 2: Mask out flat/uniform background patches (sky, plain walls)
            sim_matrix_masked[~texture_mask_flat, :] = -1.0
            sim_matrix_masked[:, ~texture_mask_flat] = -1.0

            # Filter 2: Z-Score Statistical Thresholding (> mean + 2.8 * std)
            valid_corrs = sim_matrix_masked[sim_matrix_masked > -0.99]
            if valid_corrs.numel() > 50:
                mean_corr = float(valid_corrs.mean())
                std_corr = float(valid_corrs.std())
                z_thresh = mean_corr + self.z_score_multiplier * std_corr
            else:
                mean_corr = 0.5
                std_corr = 0.1
                z_thresh = 0.995

            # Require correlation peak to exceed both Z-score and an absolute floor
            effective_threshold = float(np.clip(max(z_thresh, 0.985), 0.95, 0.9998))

            # Maximum correlation to distant textured spatial locations for each patch
            max_corr, _ = torch.max(sim_matrix_masked, dim=1)
            corr_grid = max_corr.view(hf, wf).cpu().numpy()

            # Upsample anomaly correlation heatmap to original image dimensions
            corr_map = max_corr.view(1, 1, hf, wf)
            upsampled = F.interpolate(corr_map, size=(h, w), mode="bilinear", align_corners=False)
            corr_np = upsampled.squeeze().cpu().numpy()

        max_val = float(np.max(corr_grid))
        mean_val = float(np.mean(corr_grid))

        # 4. Intermediate View: Raw Correlation Heatmap
        corr_display = np.clip((corr_np - 0.70) / 0.30 * 255.0, 0, 255).astype(np.uint8)
        colored_inferno = cv2.applyColorMap(corr_display, cv2.COLORMAP_INFERNO)
        intermediate_view = cv2.cvtColor(colored_inferno, cv2.COLOR_BGR2RGB)

        # 5. Connected Component Analysis on Z-Score Exceeded Cells
        peak_cells = (
            (corr_grid >= effective_threshold) & (corr_grid >= 0.985) & texture_mask_2d
        ).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(peak_cells, connectivity=8)

        cluster_found = False
        total_cluster_cells = 0

        for i in range(1, num_labels):
            cell_area = stats[i, cv2.CC_STAT_AREA]
            if self.min_cluster_cells <= cell_area <= int(hf * wf * 0.45):
                cluster_found = True
                total_cluster_cells += cell_area

        # 6. Forgery Decision & Localization Mask
        if cluster_found and max_val >= effective_threshold:
            is_forged = True
            confidence = float(min(99.0, max(58.0, 62.0 + total_cluster_cells * 0.5)))
        else:
            is_forged = False
            confidence = float(min(22.0, max(0.0, (max_val - 0.92) / 0.08 * 22.0)))

        binary_mask = create_empty_mask(h, w)
        if cluster_found:
            grid_mask = np.zeros((hf, wf), dtype=np.float32)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_cluster_cells:
                    grid_mask[labels == i] = 1.0

            upsampled_mask = cv2.resize(grid_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            upsampled_mask = cv2.dilate(upsampled_mask, kernel_dilate, iterations=1)
            binary_mask = cv2.GaussianBlur(upsampled_mask, (25, 25), 0)
            binary_mask = np.clip(binary_mask, 0.0, 1.0).astype(np.float32)

        visual_result = overlay_heatmap(image_rgb, binary_mask, colormap=cv2.COLORMAP_JET, alpha=0.6)

        return {
            "is_forged": is_forged,
            "confidence": confidence,
            "visual_result": visual_result,
            "intermediate_view": intermediate_view,
            "binary_mask": binary_mask,
            "metrics": {
                "max_correlation": max_val,
                "mean_correlation": mean_val,
                "z_threshold": effective_threshold,
                "exclusion_radius": exclusion_radius,
                "device": str(self.device),
                "feature_shape": [int(hf), int(wf), int(c)],
                "cluster_cells": int(total_cluster_cells),
            },
        }
