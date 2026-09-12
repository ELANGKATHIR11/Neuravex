import torch
import math

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = x.unbind(-1)
    b = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.stack(b, dim=-1)

def box_iou_2d(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute pairwise 2D IoU between boxes1 (N, 4) and boxes2 (M, 4) in xyxy format.
    Returns (N, M) tensor.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)

def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) between aligned box1 (..., 4) and box2 (..., 4) in xyxy format.
    Returns (...,) tensor of CIoU values in range [-1, 1].
    """
    # Overlap
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])

    w_inter = (x2 - x1).clamp(min=0)
    h_inter = (y2 - y1).clamp(min=0)
    inter = w_inter * h_inter

    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=0)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=0)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=0)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box
    cw = torch.max(box1[..., 2], box2[..., 2]) - torch.min(box1[..., 0], box2[..., 0])
    ch = torch.max(box1[..., 3], box2[..., 3]) - torch.min(box1[..., 1], box2[..., 1])
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    b1_cx = (box1[..., 0] + box1[..., 2]) / 2
    b1_cy = (box1[..., 1] + box1[..., 3]) / 2
    b2_cx = (box2[..., 0] + box2[..., 2]) / 2
    b2_cy = (box2[..., 1] + box2[..., 3]) / 2
    rho2 = (b2_cx - b1_cx) ** 2 + (b2_cy - b1_cy) ** 2

    # Aspect ratio consistency
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou
