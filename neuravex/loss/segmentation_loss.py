import torch
import torch.nn as nn
import torch.nn.functional as F

def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> torch.Tensor:
    """
    Multiclass Dice Loss.
    logits: (B, C, H, W) raw unnormalized logits
    target: (B, H, W) long labels in [0, C-1], with optional ignore_index
    """
    valid_mask = (target != ignore_index)
    target_clean = target.clone()
    target_clean[~valid_mask] = 0

    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    one_hot = F.one_hot(target_clean, num_classes=num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)

    # Apply valid mask across spatial dims
    valid_expanded = valid_mask.unsqueeze(1).float()
    probs = probs * valid_expanded
    one_hot = one_hot * valid_expanded

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    cardinality = (probs + one_hot).sum(dims)

    dice_score = (2.0 * intersection + eps) / (cardinality + eps)
    dice_loss = 1.0 - dice_score

    # Only average over classes present in ground truth or active
    present_classes = (one_hot.sum(dims) > 0)
    if present_classes.any():
        return dice_loss[present_classes].mean()
    return dice_loss.mean()

def semantic_segmentation_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    L_sem = CE + Multiclass Dice
    """
    num_classes = logits.shape[1]
    loss_ce = F.cross_entropy(logits, target.long(), ignore_index=ignore_index)
    loss_dice = multiclass_dice_loss(logits, target, num_classes=num_classes, ignore_index=ignore_index)
    return loss_ce + loss_dice

def boundary_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Binary focal loss on raw logits for high class imbalance in boundary maps.
    logits: (B, 1, H, W) raw logits
    target: (B, 1, H, W) float targets in [0, 1]
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * target + (1.0 - p) * (1.0 - target)
    a = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = a * (1.0 - pt).pow(gamma) * bce
    return focal.mean()

def boundary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary dice loss on raw logits for boundary map."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(-2, -1))
    union = (p + target).sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (union + eps)).mean()

def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combines boundary focal loss and boundary dice loss."""
    return boundary_focal_loss(logits, target) + boundary_dice_loss(logits, target)

def mask_quality_loss(pred_quality: torch.Tensor, pred_masks: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mask quality supervision:
    q_gt = IoU(M_p, M_g)
    L_q = SmoothL1(q_p, q_gt)
    pred_quality: (B, 1, H, W) predicted quality map (raw logits or sigmoid)
    pred_masks: (B, 1, H, W) or (B, N, H, W)
    gt_masks: (B, 1, H, W) or (B, N, H, W)
    """
    p_mask = torch.sigmoid(pred_masks) if pred_masks.dtype == torch.float32 else pred_masks.float()
    g_mask = gt_masks.float()
    inter = (p_mask * g_mask).sum(dim=(-2, -1))
    union = (p_mask + g_mask - p_mask * g_mask).sum(dim=(-2, -1))
    q_gt = (inter / (union + eps)).unsqueeze(-1).unsqueeze(-1)  # broadcastable (B, 1, 1, 1)

    q_pred = torch.sigmoid(pred_quality)
    q_gt_expanded = q_gt.expand_as(q_pred)
    return F.smooth_l1_loss(q_pred, q_gt_expanded)

def discriminative_instance_loss(embeddings: torch.Tensor, instance_ids: torch.Tensor,
                                 delta_v: float = 0.5, delta_d: float = 1.5,
                                 param_var: float = 1.0, param_dist: float = 1.0,
                                 param_reg: float = 0.001) -> torch.Tensor:
    """
    Discriminative instance embedding loss (De Brabandere et al.):
    L_inst = L_pull (variance) + L_push (distance) + L_reg
    embeddings: (B, D, H, W) normalized embedding features
    instance_ids: (B, H, W) integer instance ID map (0 = background)
    """
    B, D, H, W = embeddings.shape
    loss_total = []

    for b in range(B):
        ids = instance_ids[b].unique()
        ids = ids[ids > 0]
        num_instances = len(ids)
        if num_instances == 0:
            continue

        emb_b = embeddings[b]  # (D, H, W)
        centroids = []
        l_var = emb_b.new_tensor(0.0)

        for inst_id in ids:
            mask = (instance_ids[b] == inst_id)
            if mask.sum() < 2:
                continue
            # Pixels for this instance: (N_pixels, D)
            inst_embs = emb_b[:, mask].T
            centroid = inst_embs.mean(dim=0, keepdim=True)
            centroids.append(centroid.squeeze(0))

            # Pull: distances to centroid greater than delta_v
            dist_to_center = torch.norm(inst_embs - centroid, dim=1)
            l_var = l_var + torch.clamp(dist_to_center - delta_v, min=0.0).pow(2).mean()

        l_var = l_var / max(num_instances, 1)

        # Push: distance between centroids less than 2*delta_d
        l_dist = emb_b.new_tensor(0.0)
        if len(centroids) > 1:
            cent_stack = torch.stack(centroids, dim=0)  # (K, D)
            cent_dist = torch.cdist(cent_stack, cent_stack)  # (K, K)
            eye = torch.eye(len(centroids), device=embeddings.device, dtype=torch.bool)
            pair_dist = cent_dist[~eye]
            l_dist = torch.clamp(2.0 * delta_d - pair_dist, min=0.0).pow(2).mean()

        # Centroid regularization
        l_reg = emb_b.new_tensor(0.0)
        if len(centroids) > 0:
            cent_stack = torch.stack(centroids, dim=0)
            l_reg = torch.norm(cent_stack, dim=1).mean()

        loss_sample = param_var * l_var + param_dist * l_dist + param_reg * l_reg
        loss_total.append(loss_sample)

    if len(loss_total) > 0:
        return torch.stack(loss_total).mean()
    return embeddings.sum() * 0.0
