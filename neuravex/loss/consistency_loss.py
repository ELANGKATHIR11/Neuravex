import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformAlignedConsistencyLoss(nn.Module):
    """
    Bidirectional confidence-weighted transform-aligned multi-view consistency loss:
    L_cons = sum w_i * D(T(P), P_aug) / (sum w_i + eps)
    Preserves gradients across both branches without unilateral detach.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_base: dict, pred_aug: dict, inv_matrix: torch.Tensor, is_hflip: torch.Tensor = None) -> torch.Tensor:
        loss_total = pred_base["class_logits"].sum() * 0.0
        n_items = 0

        # 1. Dense maps
        dense_keys = ["semantic_masks", "boundary_map", "depth_map"]
        for k in dense_keys:
            if k not in pred_base or k not in pred_aug:
                continue
            base_map = pred_base[k]
            aug_map = pred_aug[k]
            B, C, H, W = base_map.shape

            if is_hflip is not None:
                aligned_aug = []
                for b in range(B):
                    m = aug_map[b:b+1]
                    if is_hflip[b]:
                        m = torch.flip(m, dims=[-1])
                    aligned_aug.append(m)
                aligned_aug = torch.cat(aligned_aug, dim=0)
            else:
                if inv_matrix.shape[-2:] == (3, 3):
                    affine_theta = inv_matrix[:, :2, :]
                else:
                    affine_theta = inv_matrix
                grid = F.affine_grid(affine_theta, [B, C, H, W], align_corners=False)
                aligned_aug = F.grid_sample(aug_map, grid, align_corners=False)

            # Bidirectional confidence weight based on entropy / confidence
            if k == "semantic_masks":
                conf_base = torch.softmax(base_map, dim=1).max(dim=1, keepdim=True).values
                conf_aug = torch.softmax(aligned_aug, dim=1).max(dim=1, keepdim=True).values
                weight = 0.5 * (conf_base + conf_aug).detach()
            else:
                weight = torch.ones_like(base_map[:, :1])

            diff = F.smooth_l1_loss(base_map, aligned_aug, reduction="none")
            weighted_diff = (diff * weight).sum() / (weight.sum() * C + self.eps)
            loss_total = loss_total + weighted_diff
            n_items += 1

        # 2. Instance embeddings
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
            loss_total = loss_total + F.smooth_l1_loss(base_emb, aligned_emb)
            n_items += 1

        # 3. Class logits consistency
        if "class_logits" in pred_base and "class_logits" in pred_aug:
            # Symmetrized KL divergence between predictive distributions
            p_base = F.log_softmax(pred_base["class_logits"], dim=-1)
            p_aug = F.log_softmax(pred_aug["class_logits"], dim=-1)
            kl_div = 0.5 * (F.kl_div(p_base, p_aug.exp(), reduction="batchmean") +
                            F.kl_div(p_aug, p_base.exp(), reduction="batchmean"))
            loss_total = loss_total + 0.1 * kl_div
            n_items += 1

        return loss_total / max(n_items, 1)
