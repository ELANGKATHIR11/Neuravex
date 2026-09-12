import torch
import torch.nn.functional as F
import random
import math

class GeometricMultiViewAugment:
    """
    Multi-view geometric augmentation tracking forward and inverse transformations:
    - Random horizontal flip
    - Random brightness/contrast jitter
    - Affine scaling / translation (optional)
    Returns:
      transformed_images: (B, C, H, W)
      inv_transforms: (B, 3, 3) matrix
      is_hflip: (B,) boolean tensor
    """
    def __init__(self, p_flip: float = 0.5, contrast_range=(0.8, 1.2)):
        self.p_flip = p_flip
        self.contrast_range = contrast_range

    def __call__(self, img: torch.Tensor):
        """
        img: (B, 3, H, W) in [0, 1]
        """
        B, C, H, W = img.shape
        device = img.device
        aug_img = img.clone()
        is_hflip = torch.zeros(B, dtype=torch.bool, device=device)
        inv_matrices = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

        for b in range(B):
            # 1. Flip
            if random.random() < self.p_flip:
                aug_img[b] = torch.flip(aug_img[b], dims=[-1])
                is_hflip[b] = True
                # Flip matrix across x: x' = -x
                flip_mat = torch.tensor([
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], device=device)
                inv_matrices[b] = flip_mat @ inv_matrices[b]

            # 2. Photometric jitter around per-image mean
            factor = random.uniform(*self.contrast_range)
            mean = aug_img[b].mean(dim=(-2, -1), keepdim=True)
            aug_img[b] = ((aug_img[b] - mean) * factor + mean).clamp(0.0, 1.0)

        return aug_img, inv_matrices, is_hflip

def invert_box_transform(boxes_xyxy: torch.Tensor, is_hflip: torch.Tensor, img_w: float) -> torch.Tensor:
    """
    Re-aligns bounding boxes after horizontal flip:
    x1_new = W - x2, x2_new = W - x1
    """
    aligned = boxes_xyxy.clone()
    for b in range(boxes_xyxy.shape[0]):
        if is_hflip[b]:
            x1 = aligned[b, ..., 0].clone()
            x2 = aligned[b, ..., 2].clone()
            aligned[b, ..., 0] = img_w - x2
            aligned[b, ..., 2] = img_w - x1
    return aligned

def invert_yaw_transform(yaw: torch.Tensor, is_hflip: torch.Tensor) -> torch.Tensor:
    """
    Re-aligns yaw rotation angles after horizontal flip:
    yaw_new = pi - yaw (or -yaw depending on coordinate convention)
    """
    aligned = yaw.clone()
    for b in range(yaw.shape[0]):
        if is_hflip[b]:
            aligned[b] = math.pi - aligned[b]
    return aligned
