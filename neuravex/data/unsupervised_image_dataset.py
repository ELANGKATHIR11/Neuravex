"""
Neuravex Self-Supervised & Slot Discovery Image Dataset.

Designed for real image datasets organized into subfolders (e.g., animals/Indian Animals/<class_name>/<img_file>)
without bounding boxes or manual annotations.

Supports:
- Recursive discovery of all image files in subdirectories
- Deterministic stratified or random Train / Val / Test splitting
- Multi-view augmentation for EMA Teacher SSL distillation
- Optional class label extraction from folder hierarchy for zero-shot / linear probing evaluation
"""

import os
import glob
import random
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UnsupervisedImageFolderDataset(Dataset):
    """
    Dataset adapter for unannotated image collections.
    Produces dual augmented views for self-supervised learning (SSL)
    along with base images for Slot Attention Object Discovery.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        img_size: int = 320,
        seed: int = 42,
        return_dual_views: bool = True,
    ):
        """
        Args:
            root_dir: Root path to image folder (e.g., 'animals' or 'animals/Indian Animals')
            split: 'train', 'val', or 'test'
            split_ratios: (train_ratio, val_ratio, test_ratio), sum to 1.0
            img_size: Target square image dimension
            seed: Random seed for deterministic splitting
            return_dual_views: If True, returns view1 and view2 for contrastive/teacher-student SSL
        """
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.split = split.lower()
        self.img_size = img_size
        self.return_dual_views = return_dual_views

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root dataset directory does not exist: {self.root_dir}")

        # 1. Collect all images and classes
        self.classes: List[str] = []
        self.samples: List[Tuple[str, int]] = []  # (path, class_idx)

        # Scan subdirectories
        subdirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        # If root contains only 1 folder (e.g., 'Indian Animals'), drill down
        if len(subdirs) == 1 and not any(
            os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))
        ):
            search_root = os.path.join(self.root_dir, subdirs[0])
            subdirs = sorted([
                d for d in os.listdir(search_root)
                if os.path.isdir(os.path.join(search_root, d))
            ])
        else:
            search_root = self.root_dir

        for cls_idx, class_name in enumerate(subdirs):
            cls_folder = os.path.join(search_root, class_name)
            img_paths = [
                os.path.join(cls_folder, f)
                for f in sorted(os.listdir(cls_folder))
                if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            ]
            if img_paths:
                self.classes.append(class_name)
                for p in img_paths:
                    self.samples.append((p, cls_idx))

        if not self.samples:
            # Fallback: flat image search
            all_imgs = [
                f for f in sorted(glob.glob(os.path.join(self.root_dir, "**", "*.*"), recursive=True))
                if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            ]
            if not all_imgs:
                raise ValueError(f"No valid image files found under: {self.root_dir}")
            self.classes = ["unlabeled"]
            self.samples = [(p, 0) for p in all_imgs]

        # 2. Partition dataset deterministically (stratified by class)
        rng = random.Random(seed)
        train_samples, val_samples, test_samples = [], [], []

        # Group by class index
        class_to_samples: Dict[int, List[Tuple[str, int]]] = {}
        for s in self.samples:
            class_to_samples.setdefault(s[1], []).append(s)

        train_r, val_r, _ = split_ratios
        for c_idx, items in class_to_samples.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(round(n * train_r))
            n_val = int(round(n * val_r))
            # Ensure at least 1 in val and test if n is large enough
            if n >= 3 and n_val == 0:
                n_val = 1
                n_train = max(1, n_train - 1)
            train_samples.extend(shuffled[:n_train])
            val_samples.extend(shuffled[n_train : n_train + n_val])
            test_samples.extend(shuffled[n_train + n_val :])

        if self.split == "train":
            self.active_samples = train_samples
        elif self.split in ("val", "valid", "validation"):
            self.active_samples = val_samples
        elif self.split == "test":
            self.active_samples = test_samples
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

    def __len__(self) -> int:
        return len(self.active_samples)

    def _read_and_resize(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            # Corrupted image fallback: return neutral gray tensor
            return np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 128
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def _augment_view(self, img_np: np.ndarray, is_eval: bool = False) -> torch.Tensor:
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        if is_eval:
            return tensor

        # Random horizontal flip
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)

        # Subtle color jitter
        if random.random() > 0.3:
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            tensor = (tensor * brightness).clamp(0.0, 1.0)

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, class_idx = self.active_samples[idx]
        base_np = self._read_and_resize(path)

        is_eval = self.split != "train"
        view1 = self._augment_view(base_np, is_eval=is_eval)

        data = {
            "image": view1,
            "path": path,
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "class_name": self.classes[class_idx] if class_idx < len(self.classes) else "unknown",
        }

        if self.return_dual_views:
            view2 = self._augment_view(base_np, is_eval=is_eval)
            data["image_aug"] = view2

        return data
