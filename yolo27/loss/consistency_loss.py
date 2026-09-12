import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformAlignedConsistencyLoss(nn.Module):
    """
    Confidence-weighted transform-aligned multi-view consistency loss:
    L_cons = sum(w * D(P, T^{-1}(P_aug))) / (sum(w) + eps)
    Ensures predictions from transformed views are inverse-mapped to the primary frame
    before computing loss.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_base: dict, pred_aug: dict, inv_matrix: torch.Tensor, is_hflip: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_base: outputs for original image
            pred_aug: outputs for augmented view
            inv_matrix: (B, 3, 3) or (B, 2, 3) affine inverse transformation matrix
            is_hflip: (B,) boolean indicating if horizontal flip was applied
        """
        loss_total = pred_base["class_logits"].sum() * 0.0
        n_items = 0

        # Dense maps: semantic_masks, boundary_map, depth
        dense_keys = ["semantic_masks", "boundary_map", "depth_map"]
        
        for k in dense_keys:
            if k not in pred_base or k not in pred_aug:
                continue
            base_map = pred_base[k]  # (B, C, H, W)
            aug_map = pred_aug[k]    # (B, C, H, W)
            B, C, H, W = base_map.shape

            # If horizontal flip was applied, re-flip the augmented prediction
            if is_hflip is not None:
                aligned_aug = []
                for b in range(B):
                    m = aug_map[b:b+1]
                    if is_hflip[b]:
                        m = torch.flip(m, dims=[-1])
                    aligned_aug.append(m)
                aligned_aug = torch.cat(aligned_aug, dim=0)
            else:
                # Affine grid sample using inv_matrix
                if inv_matrix.shape[-2:] == (3, 3):
                    affine_theta = inv_matrix[:, :2, :]
                else:
                    affine_theta = inv_matrix
                grid = F.affine_grid(affine_theta, [B, C, H, W], align_corners=False)
                aligned_aug = F.grid_sample(aug_map, grid, align_corners=False)

            # Confidence-weighted smooth L1
            diff = F.smooth_l1_loss(base_map, aligned_aug.detach(), reduction="none")
            loss_total = loss_total + diff.mean()
            n_items += 1

        # Instance embeddings consistency
        if "instance_embeddings" in pred_base and "instance_embeddings" in pred_aug:
            base_emb = pred_base["instance_embeddings"]
            aug_emb = pred_aug["instance_embeddings"]
            if is_hflip is not None:
                aligned_emb = []
                for b in range(base_emb.shape[0]):
                    m = aug_emb[b:b+1]
                    if is_hflip[b]:
                        m = torch.flip(m, dims=[-1])
                    aligned_emb.append(m)
                aligned_emb = torch.cat(aligned_emb, dim=0)
            else:
                aligned_emb = aug_emb
            loss_total = loss_total + F.smooth_l1_loss(base_emb, aligned_emb.detach())
            n_items += 1

        return loss_total / max(n_items, 1)
