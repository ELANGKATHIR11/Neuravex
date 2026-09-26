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

def box_diou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Distance-IoU (DIoU) between boxes1 (N, 4) and boxes2 (M, 4) in xyxy format.
    Returns (N, M) tensor.
    """
    # Overlap
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + eps
    iou = inter / union

    # Center distance
    c1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5  # (N, 2)
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5  # (M, 2)
    rho2 = ((c1[:, None, :] - c2[None, :, :]) ** 2).sum(dim=-1)

    # Enclosing box diagonal squared
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    c2_enc = (enc_wh[:, :, 0] ** 2 + enc_wh[:, :, 1] ** 2).clamp_min(eps)

    return iou - (rho2 / c2_enc)

def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Generalized IoU (GIoU) between boxes1 (N, 4) and boxes2 (M, 4) in xyxy format.
    Returns (N, M) tensor.
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + eps
    iou = inter / union

    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = (enc_wh[:, :, 0] * enc_wh[:, :, 1]).clamp_min(eps)

    return iou - ((enc_area - union) / enc_area)

def calculate_box_overlap_score(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Box Overlap Score / Boundary Overlap Stability (BoS) between boxes1 (N, 4) and boxes2 (M, 4).
    Combines:
      - 2D Intersection over Union (IoU)
      - Normalized Center Proximity factor exp(-rho / diag)
      - Aspect Ratio & Scale Consistency penalty
    Values range in [0.0, 1.0], where 1.0 indicates perfect spatial and boundary coincidence.
    """
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    w1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
    h1 = (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    w2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
    h2 = (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1[:, None] + area2[None, :] - inter + eps
    iou = (inter / union).clamp(0.0, 1.0)

    # Center distance & enclosing diagonal
    c1 = (boxes1[:, :2] + boxes1[:, 2:]) * 0.5
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) * 0.5
    rho2 = ((c1[:, None, :] - c2[None, :, :]) ** 2).sum(dim=-1)

    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    diag2 = (enc_wh[:, :, 0] ** 2 + enc_wh[:, :, 1] ** 2).clamp_min(eps)

    # Distance penalty: exp(-2 * sqrt(rho2 / diag2))
    dist_penalty = torch.exp(-2.0 * torch.sqrt(rho2 / diag2))

    # Scale alignment factor
    min_area = torch.min(area1[:, None], area2[None, :])
    max_area = torch.max(area1[:, None], area2[None, :]).clamp_min(eps)
    scale_factor = (min_area / max_area).clamp(0.0, 1.0)

    bos = iou * (0.65 * dist_penalty + 0.35 * scale_factor)
    return bos.clamp(0.0, 1.0)

