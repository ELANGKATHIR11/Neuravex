import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from PIL import Image

def boundary_from_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Computes exact morphological boundary target from binary mask:
    Boundary = Dilation(M) - Erosion(M)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = (dilated - eroded).astype(np.float32)
    return boundary

class COCOMultiTaskDataset(Dataset):
    """
    Complete COCO Dataset adapter for Neuravex multi-task training:
    - Detection (bounding boxes & class labels)
    - Semantic segmentation (pixel-level class maps)
    - Instance segmentation (instance ID maps & embeddings)
    - Boundary targets (morphological gradients)
    - Masking flags: depth and 3D are masked out (m_depth=0, m_3d=0) unless explicit GT is present.
    """
    def __init__(self, img_dir: str, ann_file: str, img_size: int = 320, num_classes: int = 80, transforms=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.transforms = transforms

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data.get("images", [])}
        self.img_ids = list(self.images.keys())

        # Map annotations by image ID
        self.img_to_anns = {img_id: [] for img_id in self.img_ids}
        for ann in self.coco_data.get("annotations", []):
            if ann["image_id"] in self.img_to_anns:
                self.img_to_anns[ann["image_id"]].append(ann)

        # Map category IDs to contiguous 0-indexed classes
        cats = sorted(self.coco_data.get("categories", []), key=lambda x: x["id"])
        self.cat_to_cls = {cat["id"]: idx for idx, cat in enumerate(cats[:num_classes])}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_path = os.path.join(self.img_dir, img_info["file_name"])

        # Load image
        if os.path.exists(file_path):
            img_bgr = cv2.imread(file_path)
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback for synthetic/missing image testing
            orig_h = img_info.get("height", self.img_size)
            orig_w = img_info.get("width", self.img_size)
            img_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # Scale factors to resize to (img_size, img_size)
        scale_x = self.img_size / float(orig_w)
        scale_y = self.img_size / float(orig_h)

        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Targets initialization
        semantic_map = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        instance_map = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        boundary_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        boxes = []
        labels = []

        anns = self.img_to_anns.get(img_id, [])
        for inst_idx, ann in enumerate(anns, start=1):
            if ann.get("iscrowd", 0) == 1:
                continue
            cat_id = ann.get("category_id")
            if cat_id not in self.cat_to_cls:
                continue
            cls_id = self.cat_to_cls[cat_id]

            # 2D Bounding Box [x, y, w, h] -> [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            x1 = max(0, min(self.img_size - 1, x * scale_x))
            y1 = max(0, min(self.img_size - 1, y * scale_y))
            x2 = max(0, min(self.img_size - 1, (x + w) * scale_x))
            y2 = max(0, min(self.img_size - 1, (y + h) * scale_y))

            if (x2 - x1) > 1 and (y2 - y1) > 1:
                boxes.append([x1, y1, x2, y2])
                labels.append([cls_id])

            # Segmentation polygon rasterization
            seg = ann.get("segmentation")
            if isinstance(seg, list):
                poly_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for p in seg:
                    poly_pts = np.array(p, dtype=np.float32).reshape(-1, 2)
                    poly_pts = np.int32([poly_pts])
                    cv2.fillPoly(poly_mask, poly_pts, 1)

                resized_mask = cv2.resize(poly_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                inst_pixels = (resized_mask == 1)

                semantic_map[inst_pixels] = cls_id
                instance_map[inst_pixels] = inst_idx
                b_target = boundary_from_binary_mask(resized_mask)
                boundary_map = np.maximum(boundary_map, b_target)

        # Convert to tensors
        if len(boxes) > 0:
            gt_boxes = torch.tensor(boxes, dtype=torch.float32)
            gt_labels = torch.tensor(labels, dtype=torch.long)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0, 1), dtype=torch.long)

        return {
            "image": img_tensor,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "semantic_mask": torch.from_numpy(semantic_map).long(),
            "instance_mask": torch.from_numpy(instance_map).long(),
            "boundary_map": torch.from_numpy(boundary_map).unsqueeze(0).float(),
            # Explicit modality masking: depth and 3D are inactive on pure COCO
            "task_masks": {
                "det": 1.0,
                "semantic": 1.0,
                "instance": 1.0,
                "boundary": 1.0,
                "mask_quality": 1.0,
                "depth": 0.0,
                "geometry_3d": 0.0,
                "consistency": 1.0
            }
        }

def coco_collate_fn(batch):
    """
    Custom collator padding variable length ground-truth bounding boxes per batch.
    """
    imgs = torch.stack([item["image"] for item in batch], dim=0)
    sem_masks = torch.stack([item["semantic_mask"] for item in batch], dim=0)
    inst_masks = torch.stack([item["instance_mask"] for item in batch], dim=0)
    bound_maps = torch.stack([item["boundary_map"] for item in batch], dim=0)

    # Pad boxes and labels to max_gt
    max_gt = max([item["gt_boxes"].shape[0] for item in batch] + [1])
    B = len(batch)
    padded_boxes = torch.zeros((B, max_gt, 4), dtype=torch.float32)
    padded_labels = torch.zeros((B, max_gt, 1), dtype=torch.long)
    mask_gt = torch.zeros((B, max_gt, 1), dtype=torch.bool)

    for b, item in enumerate(batch):
        n = item["gt_boxes"].shape[0]
        if n > 0:
            padded_boxes[b, :n] = item["gt_boxes"]
            padded_labels[b, :n] = item["gt_labels"]
            mask_gt[b, :n] = True

    task_masks = batch[0]["task_masks"]

    return {
        "images": imgs,
        "gt_boxes": padded_boxes,
        "gt_labels": padded_labels,
        "mask_gt": mask_gt,
        "semantic_masks": sem_masks,
        "instance_masks": inst_masks,
        "boundary_maps": bound_maps,
        "task_masks": task_masks
    }
