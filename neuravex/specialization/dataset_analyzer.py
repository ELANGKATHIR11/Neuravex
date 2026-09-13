"""
Neuravex Dataset Analyzer.

Analyzes any COCO-format dataset and extracts a normalized dataset-complexity vector
used by the Architecture Generator for dataset-conditioned model specialization.

Complexity vector dimensions (16):
  [0]  num_classes_norm         - Normalized class count
  [1]  class_imbalance          - Gini coefficient of class distribution
  [2]  small_obj_fraction       - % objects with area < 32^2 (COCO APs)
  [3]  medium_obj_fraction      - % objects with 32^2 <= area < 96^2
  [4]  large_obj_fraction       - % objects with area >= 96^2
  [5]  mean_obj_density         - Mean objects per image (normalized)
  [6]  density_variance         - Variance of object counts across images (normalized)
  [7]  mean_aspect_ratio_dev    - Mean deviation from square aspect ratio
  [8]  resolution_norm          - Mean image resolution (normalized to 1280px)
  [9]  texture_complexity       - Mean Laplacian variance (scene complexity) normalized
  [10] foreground_ratio         - Mean foreground pixel fraction
  [11] occlusion_proxy          - Estimated occlusion via bbox overlap proxy
  [12] class_difficulty_spread  - Std of per-class instance counts (harder = unequal)
  [13] scene_diversity          - Estimated scene diversity proxy
  [14] multi_label_rate         - Fraction of images with overlapping annotations
  [15] label_density            - Total annotations / total images (normalized)
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class DatasetStats:
    """Structured statistics extracted from a dataset."""
    num_classes: int = 0
    class_counts: Dict[int, int] = field(default_factory=dict)
    total_images: int = 0
    total_annotations: int = 0
    object_areas: List[float] = field(default_factory=list)
    object_aspect_ratios: List[float] = field(default_factory=list)
    objects_per_image: List[int] = field(default_factory=list)
    image_resolutions: List[Tuple[int, int]] = field(default_factory=list)
    texture_complexities: List[float] = field(default_factory=list)
    foreground_ratios: List[float] = field(default_factory=list)
    occlusion_proxies: List[float] = field(default_factory=list)
    # Derived
    complexity_vector: Optional[np.ndarray] = None


class DatasetAnalyzer:
    """
    Analyzes a COCO-format dataset to extract a normalized complexity vector
    used for architecture specialization.

    Usage:
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze(ann_file, img_dir=None, max_images=None)
        vec = stats.complexity_vector  # shape (16,) in [0, 1]
        summary = analyzer.summarize(stats)
    """

    VECTOR_DIM = 16
    # COCO area thresholds
    SMALL_THRESH = 32 ** 2    # 1024
    LARGE_THRESH = 96 ** 2    # 9216

    def __init__(self, max_classes: int = 1000, max_density: float = 100.0,
                 max_resolution: int = 1280, max_texture: float = 5000.0,
                 max_label_density: float = 50.0):
        self.max_classes = max_classes
        self.max_density = max_density
        self.max_resolution = max_resolution
        self.max_texture = max_texture
        self.max_label_density = max_label_density

    def analyze(self, ann_file: str, img_dir: Optional[str] = None,
                max_images: Optional[int] = None) -> DatasetStats:
        """Analyze a COCO-format annotation file and return DatasetStats."""
        with open(ann_file, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco.get("images", [])}
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        num_classes = len(categories)
        cat_ids = [c["id"] for c in categories]

        # Organize annotations by image
        img_anns = {img_id: [] for img_id in images}
        for ann in annotations:
            if ann.get("image_id") in img_anns:
                img_anns[ann["image_id"]].append(ann)

        stats = DatasetStats()
        stats.num_classes = num_classes
        stats.total_images = len(images)
        stats.total_annotations = len(annotations)

        # Class distribution
        class_counts = {cat_id: 0 for cat_id in cat_ids}
        for ann in annotations:
            cid = ann.get("category_id")
            if cid in class_counts:
                class_counts[cid] += 1
        stats.class_counts = class_counts

        img_ids = list(images.keys())
        if max_images is not None:
            img_ids = img_ids[:max_images]

        for img_id in img_ids:
            img_info = images[img_id]
            anns_for_img = img_anns.get(img_id, [])
            img_h = img_info.get("height", 640)
            img_w = img_info.get("width", 640)
            img_area = img_h * img_w

            stats.image_resolutions.append((img_h, img_w))
            stats.objects_per_image.append(len(anns_for_img))

            fg_area = 0.0
            boxes = []
            for ann in anns_for_img:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                area = w * h
                stats.object_areas.append(area)
                aspect = w / max(h, 1e-4)
                stats.object_aspect_ratios.append(aspect)
                fg_area += area
                boxes.append([x, y, x + w, y + h])

            # Foreground ratio
            fg_ratio = min(fg_area / max(img_area, 1e-4), 1.0)
            stats.foreground_ratios.append(fg_ratio)

            # Occlusion proxy: compute pairwise IoU among boxes in image
            if len(boxes) >= 2:
                occ_proxy = self._compute_occlusion_proxy(boxes, img_area)
            else:
                occ_proxy = 0.0
            stats.occlusion_proxies.append(occ_proxy)

            # Texture complexity via Laplacian variance if cv2 available and img_dir given
            if img_dir and HAS_CV2:
                fname = img_info.get("file_name", "")
                fpath = os.path.join(img_dir, fname)
                if os.path.exists(fpath):
                    try:
                        img_bgr = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                        if img_bgr is not None:
                            lap_var = float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())
                            stats.texture_complexities.append(lap_var)
                    except Exception:
                        pass

        # Build complexity vector
        stats.complexity_vector = self._build_complexity_vector(stats)
        return stats

    def _compute_occlusion_proxy(self, boxes: List[List[float]], img_area: float) -> float:
        """Compute mean pairwise IoU as occlusion proxy."""
        n = len(boxes)
        if n < 2:
            return 0.0
        total_iou = 0.0
        count = 0
        # Limit to first 20 boxes per image for speed
        boxes = boxes[:20]
        n = len(boxes)
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._box_iou(boxes[i], boxes[j])
                total_iou += iou
                count += 1
        return total_iou / max(count, 1)

    @staticmethod
    def _box_iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        union = a_area + b_area - inter
        return inter / max(union, 1e-6)

    @staticmethod
    def _gini(counts: List[int]) -> float:
        """Compute Gini coefficient of count distribution."""
        if not counts or sum(counts) == 0:
            return 0.0
        arr = sorted(counts)
        n = len(arr)
        cumsum = sum(arr[i] * (2 * (i + 1) - n - 1) for i in range(n))
        return cumsum / max(n * sum(arr), 1e-6)

    def _build_complexity_vector(self, stats: DatasetStats) -> np.ndarray:
        """Build normalized 16-dim complexity vector from stats."""
        vec = np.zeros(self.VECTOR_DIM, dtype=np.float32)

        # [0] Normalized class count
        vec[0] = min(stats.num_classes / self.max_classes, 1.0)

        # [1] Class imbalance (Gini)
        counts = list(stats.class_counts.values())
        vec[1] = abs(self._gini(counts))

        # [2-4] Object size distribution
        areas = np.array(stats.object_areas) if stats.object_areas else np.zeros(1)
        if len(areas) > 0:
            small = float((areas < self.SMALL_THRESH).sum()) / len(areas)
            large = float((areas >= self.LARGE_THRESH).sum()) / len(areas)
            medium = 1.0 - small - large
            vec[2] = float(small)
            vec[3] = float(max(medium, 0.0))
            vec[4] = float(large)

        # [5] Mean object density (normalized)
        density = np.array(stats.objects_per_image) if stats.objects_per_image else np.zeros(1)
        vec[5] = min(float(density.mean()), self.max_density) / self.max_density

        # [6] Density variance (normalized)
        vec[6] = min(float(density.std()), self.max_density) / self.max_density

        # [7] Mean aspect ratio deviation from square
        if stats.object_aspect_ratios:
            aspects = np.array(stats.object_aspect_ratios)
            vec[7] = min(float(np.abs(aspects - 1.0).mean()), 5.0) / 5.0

        # [8] Resolution (normalized)
        if stats.image_resolutions:
            mean_res = np.mean([max(h, w) for h, w in stats.image_resolutions])
            vec[8] = min(float(mean_res), self.max_resolution) / self.max_resolution

        # [9] Texture complexity
        if stats.texture_complexities:
            mean_tex = np.mean(stats.texture_complexities)
            vec[9] = min(float(mean_tex), self.max_texture) / self.max_texture
        else:
            vec[9] = 0.5  # Unknown → assume moderate

        # [10] Foreground ratio
        if stats.foreground_ratios:
            vec[10] = float(np.mean(stats.foreground_ratios))

        # [11] Occlusion proxy
        if stats.occlusion_proxies:
            vec[11] = float(np.clip(np.mean(stats.occlusion_proxies) * 5.0, 0.0, 1.0))

        # [12] Class difficulty spread (normalized std of per-class counts)
        if counts:
            total = max(sum(counts), 1)
            normed = [c / total for c in counts]
            vec[12] = min(float(np.std(normed)) * 20.0, 1.0)

        # [13] Scene diversity proxy (variance of resolutions)
        if len(stats.image_resolutions) > 1:
            res_vals = [max(h, w) for h, w in stats.image_resolutions]
            vec[13] = min(float(np.std(res_vals)) / 640.0, 1.0)

        # [14] Multi-label rate (images with > mean annotations)
        if stats.objects_per_image:
            mean_obj = float(density.mean())
            multi = float(sum(c > mean_obj * 1.5 for c in stats.objects_per_image))
            vec[14] = min(multi / max(stats.total_images, 1), 1.0)

        # [15] Label density (annotations/image normalized)
        if stats.total_images > 0:
            label_density = stats.total_annotations / stats.total_images
            vec[15] = min(label_density / self.max_label_density, 1.0)

        return vec

    def summarize(self, stats: DatasetStats) -> Dict:
        """Generate a human-readable summary of dataset statistics."""
        areas = np.array(stats.object_areas) if stats.object_areas else np.zeros(1)
        small_frac = float((areas < self.SMALL_THRESH).sum()) / max(len(areas), 1)
        med_frac = float(((areas >= self.SMALL_THRESH) & (areas < self.LARGE_THRESH)).sum()) / max(len(areas), 1)
        large_frac = float((areas >= self.LARGE_THRESH).sum()) / max(len(areas), 1)

        return {
            "num_classes": stats.num_classes,
            "total_images": stats.total_images,
            "total_annotations": stats.total_annotations,
            "annotations_per_image": round(stats.total_annotations / max(stats.total_images, 1), 2),
            "class_imbalance_gini": round(abs(self._gini(list(stats.class_counts.values()))), 4),
            "object_size_fractions": {
                "small (<32^2)": round(small_frac, 4),
                "medium": round(med_frac, 4),
                "large (>=96^2)": round(large_frac, 4),
            },
            "mean_objects_per_image": round(float(np.mean(stats.objects_per_image)) if stats.objects_per_image else 0, 2),
            "mean_foreground_ratio": round(float(np.mean(stats.foreground_ratios)) if stats.foreground_ratios else 0, 4),
            "mean_occlusion_proxy": round(float(np.mean(stats.occlusion_proxies)) if stats.occlusion_proxies else 0, 4),
            "complexity_vector": stats.complexity_vector.tolist() if stats.complexity_vector is not None else [],
            "recommended_scale_emphasis": self._recommend_scales(stats.complexity_vector),
        }

    def _recommend_scales(self, vec: Optional[np.ndarray]) -> str:
        if vec is None:
            return "unknown"
        small_frac = vec[2]
        large_frac = vec[4]
        if small_frac > 0.5:
            return "P2+P3 (tiny-object emphasis)"
        elif large_frac > 0.5:
            return "P5 (large-object emphasis)"
        elif small_frac > 0.3:
            return "P3+P4 (mixed-small)"
        else:
            return "P3+P4+P5 (balanced)"
