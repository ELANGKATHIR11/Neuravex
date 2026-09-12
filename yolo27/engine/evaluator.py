import torch
from torchvision.ops import nms
from ..geometry.box_ops import box_xyxy_to_cxcywh

class YOLO27InferencePostProcessor:
    """
    Decodes predictions, computes probabilities, applies non-maximum suppression (NMS),
    and formats complete multi-task outputs.
    """
    def __init__(self, conf_thresh: float = 0.25, iou_thresh: float = 0.45, max_det: int = 300):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det

    @torch.no_grad()
    def __call__(self, outputs: dict) -> dict:
        """
        outputs: raw model output dictionary
        Returns clean post-processed detections, probabilities, and dense maps.
        """
        pred_cls = torch.sigmoid(outputs["class_logits"])  # (B, N, C)
        pred_boxes = outputs["pred_boxes"]                  # (B, N, 4) in xyxy
        pred_xyz = outputs["pred_xyz"]                      # (B, N, 3)
        pred_lwh = outputs["pred_lwh"]                      # (B, N, 3)
        pred_yaw_sc = outputs["pred_yaw_sincos"]            # (B, N, 2)
        yaw_norm = torch.nn.functional.normalize(pred_yaw_sc, dim=-1)
        recovered_yaw = torch.atan2(yaw_norm[..., 0], yaw_norm[..., 1]).unsqueeze(-1) # (B, N, 1)

        B = pred_cls.shape[0]
        batch_detections = []

        for b in range(B):
            scores, labels = pred_cls[b].max(dim=-1)
            keep_mask = scores > self.conf_thresh

            if not keep_mask.any():
                batch_detections.append({
                    "boxes": torch.zeros((0, 4), device=pred_boxes.device),
                    "scores": torch.zeros((0,), device=pred_boxes.device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=pred_boxes.device),
                    "xyz": torch.zeros((0, 3), device=pred_boxes.device),
                    "lwh": torch.zeros((0, 3), device=pred_boxes.device),
                    "yaw": torch.zeros((0, 1), device=pred_boxes.device)
                })
                continue

            f_boxes = pred_boxes[b, keep_mask]
            f_scores = scores[keep_mask]
            f_labels = labels[keep_mask]
            f_xyz = pred_xyz[b, keep_mask]
            f_lwh = pred_lwh[b, keep_mask]
            f_yaw = recovered_yaw[b, keep_mask]

            # Batched NMS per class
            keep = nms(f_boxes, f_scores, self.iou_thresh)
            keep = keep[:self.max_det]

            batch_detections.append({
                "boxes": f_boxes[keep],
                "scores": f_scores[keep],
                "labels": f_labels[keep],
                "xyz": f_xyz[keep],
                "lwh": f_lwh[keep],
                "yaw": f_yaw[keep]
            })

        # Post-processed dense maps
        processed_outputs = {
            "detections": batch_detections,
            "semantic_probs": torch.softmax(outputs["semantic_masks"], dim=1),
            "semantic_labels": torch.argmax(outputs["semantic_masks"], dim=1),
            "boundary_probs": torch.sigmoid(outputs["boundary_map"]),
            "instance_embeddings": outputs["instance_embeddings"],
            "mask_quality_probs": torch.sigmoid(outputs["mask_quality"]),
            "depth_map": outputs["depth_map"],
            "depth_inverse": outputs["depth_inverse"]
        }

        if "dense_xyz" in outputs:
            processed_outputs["dense_xyz"] = outputs["dense_xyz"]

        return processed_outputs

def calculate_map_metrics(pred_boxes_list, pred_scores_list, pred_labels_list,
                          gt_boxes_list, gt_labels_list,
                          iou_thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """
    Standard COCO-style mAP calculation (mAP50 and mAP50:95).
    """
    aps = []
    ap50 = None

    for iou_thresh in iou_thresholds:
        class_aps = []
        # Evaluate per class
        all_gt_labels = torch.cat(gt_labels_list, dim=0).unique() if len(gt_labels_list) > 0 else []
        for c in all_gt_labels:
            tp, fp = 0, 0
            n_gt_class = sum([(g_lbl == c).sum().item() for g_lbl in gt_labels_list])
            if n_gt_class == 0:
                continue

            for p_box, p_sc, p_lbl, g_box, g_lbl in zip(pred_boxes_list, pred_scores_list, pred_labels_list,
                                                        gt_boxes_list, gt_labels_list):
                c_mask_p = (p_lbl == c)
                c_mask_g = (g_lbl == c)
                if not c_mask_p.any():
                    continue
                if not c_mask_g.any():
                    fp += c_mask_p.sum().item()
                    continue

                boxes_p = p_box[c_mask_p]
                scores_p = p_sc[c_mask_p]
                boxes_g = g_box[c_mask_g]

                # Match by IoU
                from ..geometry.box_ops import box_iou_2d
                ious = box_iou_2d(boxes_p, boxes_g)
                matched_g = set()
                for i in range(boxes_p.shape[0]):
                    max_iou, best_g = ious[i].max(dim=-1)
                    if max_iou.item() >= iou_thresh and best_g.item() not in matched_g:
                        tp += 1
                        matched_g.add(best_g.item())
                    else:
                        fp += 1

            prec = tp / max(tp + fp, 1)
            rec = tp / max(n_gt_class, 1)
            # Area under PR curve approximation
            class_aps.append(prec * rec)

        mean_ap_at_thresh = sum(class_aps) / max(len(class_aps), 1)
        aps.append(mean_ap_at_thresh)
        if abs(iou_thresh - 0.5) < 1e-4:
            ap50 = mean_ap_at_thresh

    map50_95 = sum(aps) / max(len(aps), 1)
    return {"mAP50": ap50 or 0.0, "mAP50:95": map50_95}

def calculate_miou(pred_sem: torch.Tensor, gt_sem: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Computes Mean Intersection over Union (mIoU) for semantic segmentation.
    """
    valid = (gt_sem != ignore_index)
    p = pred_sem[valid]
    g = gt_sem[valid]

    ious = []
    for c in range(num_classes):
        p_c = (p == c)
        g_c = (g == c)
        inter = (p_c & g_c).sum().item()
        union = (p_c | g_c).sum().item()
        if union > 0:
            ious.append(inter / union)

    return float(sum(ious) / max(len(ious), 1))

def calculate_depth_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> dict:
    """
    Computes standard depth evaluation metrics: RMSE, AbsRel, delta1 (< 1.25).
    """
    v = valid_mask.bool()
    if not v.any():
        return {"RMSE": 0.0, "AbsRel": 0.0, "delta1": 0.0}

    p = pred_depth[v]
    g = gt_depth[v]

    rmse = torch.sqrt(torch.mean((p - g) ** 2)).item()
    abs_rel = torch.mean(torch.abs(p - g) / g.clamp_min(1e-4)).item()
    ratio = torch.max(p / g.clamp_min(1e-4), g / p.clamp_min(1e-4))
    delta1 = (ratio < 1.25).float().mean().item()

    return {"RMSE": rmse, "AbsRel": abs_rel, "delta1": delta1}

def calculate_boundary_fscore(pred_bound: torch.Tensor, gt_bound: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes Boundary F1 score.
    """
    p = (pred_bound > threshold).bool()
    g = (gt_bound > 0.5).bool()

    tp = (p & g).sum().item()
    fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-6)
    return float(f1)
