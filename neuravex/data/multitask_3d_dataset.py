import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class Multitask3DDataset(Dataset):
    """
    Dataset adapter for 3D Bounding Box (X, Y, Z, L, W, H, yaw) and DEM Depth supervision.
    Loads real physical metric geometry, camera intrinsics, and dense depth maps.
    """
    def __init__(self, data_dir="data/synthetic_3d_dataset", split="train", img_size=320, num_classes=4):
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.lbl_dir = os.path.join(self.split_dir, "labels_3d")
        self.img_size = img_size
        self.num_classes = num_classes

        self.samples = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.lbl_dir) if f.endswith(".json")
        ])

        with open(os.path.join(data_dir, "intrinsics.json"), "r") as f:
            self.intrinsics = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # 1. Load RGB Image
        img_path = os.path.join(self.img_dir, f"{sample_id}.jpg")
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        # 2. Load Metric Depth Map
        depth_path = os.path.join(self.depth_dir, f"{sample_id}.npy")
        depth_np = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        valid_depth = (depth_tensor > 0.1) & (depth_tensor < 60.0)

        # 3. Load 3D and 2D Annotations
        lbl_path = os.path.join(self.lbl_dir, f"{sample_id}.json")
        with open(lbl_path, "r") as f:
            data = json.load(f)

        boxes_2d = []
        labels = []
        boxes_3d_xyz = []
        boxes_3d_lwh = []
        boxes_3d_yaw = []

        sem_map = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        inst_map = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        bound_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for inst_idx, ann in enumerate(data.get("annotations", []), start=1):
            x1, y1, x2, y2 = ann["bbox_2d"]
            cls_id = ann["class_id"]

            boxes_2d.append([x1, y1, x2, y2])
            labels.append([cls_id])
            boxes_3d_xyz.append(ann["center_3d"])
            boxes_3d_lwh.append(ann["lwh_3d"])
            boxes_3d_yaw.append([ann["yaw"]])

            x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
            sem_map[y1_i:y2_i, x1_i:x2_i] = cls_id + 1
            inst_map[y1_i:y2_i, x1_i:x2_i] = inst_idx

            # Morphological boundary
            bound_map[y1_i:min(y1_i+2, self.img_size), x1_i:x2_i] = 1.0
            bound_map[max(0, y2_i-2):y2_i, x1_i:x2_i] = 1.0
            bound_map[y1_i:y2_i, x1_i:min(x1_i+2, self.img_size)] = 1.0
            bound_map[y1_i:y2_i, max(0, x2_i-2):x2_i] = 1.0

        if len(boxes_2d) > 0:
            gt_boxes = torch.tensor(boxes_2d, dtype=torch.float32)
            gt_labels = torch.tensor(labels, dtype=torch.long)
            gt_xyz = torch.tensor(boxes_3d_xyz, dtype=torch.float32)
            gt_lwh = torch.tensor(boxes_3d_lwh, dtype=torch.float32)
            gt_yaw = torch.tensor(boxes_3d_yaw, dtype=torch.float32)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0, 1), dtype=torch.long)
            gt_xyz = torch.zeros((0, 3), dtype=torch.float32)
            gt_lwh = torch.zeros((0, 3), dtype=torch.float32)
            gt_yaw = torch.zeros((0, 1), dtype=torch.float32)

        return {
            "image": img_tensor,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_3d_xyz": gt_xyz,
            "gt_3d_lwh": gt_lwh,
            "gt_3d_yaw": gt_yaw,
            "depth": depth_tensor,
            "valid_depth": valid_depth,
            "semantic_mask": torch.from_numpy(sem_map).long(),
            "instance_mask": torch.from_numpy(inst_map).long(),
            "boundary_map": torch.from_numpy(bound_map).unsqueeze(0).float(),
            # Full multi-task supervision active: 2D, 3D, Depth/DEM, Segments
            "task_masks": {
                "det": 1.0,
                "semantic": 1.0,
                "instance": 1.0,
                "boundary": 1.0,
                "mask_quality": 1.0,
                "depth": 1.0,
                "geometry_3d": 1.0,
                "consistency": 1.0
            }
        }

def multitask_3d_collate_fn(batch):
    imgs = torch.stack([item["image"] for item in batch], dim=0)
    depths = torch.stack([item["depth"] for item in batch], dim=0)
    valid_depths = torch.stack([item["valid_depth"] for item in batch], dim=0)
    sem_masks = torch.stack([item["semantic_mask"] for item in batch], dim=0)
    inst_masks = torch.stack([item["instance_mask"] for item in batch], dim=0)
    bound_maps = torch.stack([item["boundary_map"] for item in batch], dim=0)

    max_gt = max([item["gt_boxes"].shape[0] for item in batch] + [1])
    B = len(batch)

    padded_boxes = torch.zeros((B, max_gt, 4), dtype=torch.float32)
    padded_labels = torch.zeros((B, max_gt, 1), dtype=torch.long)
    padded_xyz = torch.zeros((B, max_gt, 3), dtype=torch.float32)
    padded_lwh = torch.zeros((B, max_gt, 3), dtype=torch.float32)
    padded_yaw = torch.zeros((B, max_gt, 1), dtype=torch.float32)
    mask_gt = torch.zeros((B, max_gt, 1), dtype=torch.bool)

    for b, item in enumerate(batch):
        n = item["gt_boxes"].shape[0]
        if n > 0:
            padded_boxes[b, :n] = item["gt_boxes"]
            padded_labels[b, :n] = item["gt_labels"]
            padded_xyz[b, :n] = item["gt_3d_xyz"]
            padded_lwh[b, :n] = item["gt_3d_lwh"]
            padded_yaw[b, :n] = item["gt_3d_yaw"]
            mask_gt[b, :n] = True

    return {
        "images": imgs,
        "gt_boxes": padded_boxes,
        "gt_labels": padded_labels,
        "gt_3d_xyz": padded_xyz,
        "gt_3d_lwh": padded_lwh,
        "gt_3d_yaw": padded_yaw,
        "mask_gt": mask_gt,
        "depth": depths,
        "valid_depth": valid_depths,
        "semantic_masks": sem_masks,
        "instance_masks": inst_masks,
        "boundary_maps": bound_maps,
        "task_masks": batch[0]["task_masks"]
    }
