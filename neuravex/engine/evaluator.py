from typing import Any, Dict, List, Optional, Union, cast
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import batched_nms

class NeuravexInferencePostProcessor:
    """
    Decodes predictions, computes probabilities, applies class-aware batched NMS,
    and formats complete multi-task outputs.
    """
    def __init__(self, conf_thresh: float = 0.25, iou_thresh: float = 0.45, max_det: int = 300):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.max_det = max_det

    @torch.no_grad()
    def __call__(self, outputs: dict, intrinsics=None, tracker=None, dt: float = 1.0 / 30.0) -> dict:
        pred_cls = torch.sigmoid(outputs["class_logits"])  # (B, N, C)
        pred_boxes = outputs["pred_boxes"]                  # (B, N, 4) in xyxy
        pred_xyz = outputs.get("pred_xyz", None)
        pred_lwh = outputs.get("pred_lwh", None)
        pred_yaw_sc = outputs.get("pred_yaw_sincos", None)
        if pred_yaw_sc is not None:
            yaw_norm = F.normalize(pred_yaw_sc, dim=-1)
            recovered_yaw = torch.atan2(yaw_norm[..., 0], yaw_norm[..., 1]).unsqueeze(-1)
        else:
            recovered_yaw = None

        B = pred_cls.shape[0]
        batch_detections = []
        batch_objects = []

        depth_map = outputs.get("depth_map", None)
        depth_conf = outputs.get("depth_confidence", None)
        semantic_masks = outputs.get("semantic_masks", None)
        inst_embeddings = outputs.get("instance_embeddings", None)

        from .tracker import robust_mask_depth_estimator

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
                batch_objects.append([])
                continue

            f_boxes = pred_boxes[b, keep_mask]
            f_scores = scores[keep_mask]
            f_labels = labels[keep_mask]
            f_xyz = pred_xyz[b, keep_mask] if pred_xyz is not None else torch.zeros((len(f_boxes), 3), device=f_boxes.device)
            f_lwh = pred_lwh[b, keep_mask] if pred_lwh is not None else torch.zeros((len(f_boxes), 3), device=f_boxes.device)
            f_yaw = recovered_yaw[b, keep_mask] if recovered_yaw is not None else torch.zeros((len(f_boxes), 1), device=f_boxes.device)

            # True class-aware batched NMS
            keep = batched_nms(f_boxes, f_scores, f_labels, self.iou_thresh)
            keep = keep[:self.max_det]

            det_boxes = f_boxes[keep]
            det_scores = f_scores[keep]
            det_labels = f_labels[keep]
            det_xyz = f_xyz[keep]
            det_lwh = f_lwh[keep]
            det_yaw = f_yaw[keep]

            # Bounding Box Voting (IoU-Weighted BBox Fusion):
            # Eliminates sub-pixel anchor quantization noise, tightening box boundaries and driving IoU and BoS higher
            if len(det_boxes) > 0 and len(f_boxes) > len(det_boxes):
                from ..geometry.box_ops import box_iou_2d
                ious = box_iou_2d(det_boxes, f_boxes)  # (N_keep, N_all)
                refined_boxes = []
                for k in range(len(det_boxes)):
                    same_class = (f_labels == det_labels[k])
                    match_mask = same_class & (ious[k] >= 0.50)
                    if match_mask.any():
                        match_b = f_boxes[match_mask]
                        match_s = f_scores[match_mask] * ious[k, match_mask]
                        w_sum = match_s.sum().clamp_min(1e-6)
                        voted_b = (match_b * match_s.unsqueeze(-1)).sum(dim=0) / w_sum
                        refined_boxes.append(voted_b)
                    else:
                        refined_boxes.append(det_boxes[k])
                det_boxes = torch.stack(refined_boxes, dim=0)

            batch_detections.append({
                "boxes": det_boxes,
                "scores": det_scores,
                "labels": det_labels,
                "xyz": det_xyz,
                "lwh": det_lwh,
                "yaw": det_yaw
            })

            # Per-object metric extraction (mask-depth fusion + XYZ + confidence)
            frame_objects = []
            cur_depth = depth_map[b] if depth_map is not None else None
            cur_conf = depth_conf[b] if depth_conf is not None else None

            # Proto masks and mask coefficients for YOLACT-style instance mask assembly
            proto_b = outputs.get("proto_masks")
            mask_coeffs_raw = outputs.get("pred_mask_coeffs")

            # Pre-compute instance masks for all detections in this batch
            instance_masks_b = []
            if proto_b is not None and mask_coeffs_raw is not None and len(det_boxes) > 0:
                protos = proto_b[b]  # (num_proto, Hp, Wp)
                # Get mask coefficients for kept detections
                all_coeffs = mask_coeffs_raw[b, keep_mask]
                kept_coeffs = all_coeffs[keep]  # (N_det, num_proto)
                Hp, Wp = protos.shape[1], protos.shape[2]
                # mask_logits = coeffs @ protos → (N_det, Hp, Wp)
                mask_logits = torch.einsum("nc,chw->nhw", kept_coeffs, protos)
                mask_probs_all = torch.sigmoid(mask_logits)
                for k in range(len(det_boxes)):
                    instance_masks_b.append(mask_probs_all[k])

            for i in range(len(det_boxes)):
                box = det_boxes[i]
                score = float(det_scores[i].item())
                label = int(det_labels[i].item())
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())

                # TRUE instance mask: proto_assembly or box_fallback
                obj_mask = None
                mask_source = "none"
                if cur_depth is not None:
                    _, H_d, W_d = cur_depth.shape if cur_depth.ndim == 3 else (1, cur_depth.shape[0], cur_depth.shape[1])

                    if i < len(instance_masks_b):
                        # YOLACT-style: resize proto mask to depth map resolution, crop to bbox
                        proto_mask = instance_masks_b[i]  # (Hp, Wp) probabilities
                        proto_resized = F.interpolate(
                            proto_mask.unsqueeze(0).unsqueeze(0),
                            size=(H_d, W_d), mode="bilinear", align_corners=False,
                        ).squeeze()
                        # Crop to bbox to prevent cross-object leakage
                        obj_mask = torch.zeros((H_d, W_d), dtype=torch.bool, device=box.device)
                        x1_c = max(0, min(W_d - 1, x1))
                        x2_c = max(0, min(W_d, x2))
                        y1_c = max(0, min(H_d - 1, y1))
                        y2_c = max(0, min(H_d, y2))
                        if x2_c > x1_c and y2_c > y1_c:
                            obj_mask[y1_c:y2_c, x1_c:x2_c] = (proto_resized[y1_c:y2_c, x1_c:x2_c] > 0.5)
                        mask_source = "proto_assembly"
                    else:
                        # Fallback: box region (explicitly marked, not pretending to be instance mask)
                        obj_mask = torch.zeros((H_d, W_d), dtype=torch.bool, device=box.device)
                        x1_c = max(0, min(W_d - 1, x1))
                        x2_c = max(0, min(W_d, x2))
                        y1_c = max(0, min(H_d - 1, y1))
                        y2_c = max(0, min(H_d, y2))
                        if x2_c > x1_c and y2_c > y1_c:
                            obj_mask[y1_c:y2_c, x1_c:x2_c] = True
                        mask_source = "box_fallback"

                    # Robust confidence-weighted trimmed/median depth
                    z_est, u_c, v_c, d_conf = robust_mask_depth_estimator(
                        cur_depth, obj_mask, cur_conf
                    )

                    # Compute camera XYZ:
                    # Z = Zobj; X = (u - cx)*Z/fx; Y = (v - cy)*Z/fy; R = sqrt(X^2 + Y^2 + Z^2)
                    if intrinsics is not None:
                        fx = float(intrinsics.fx)
                        fy = float(intrinsics.fy)
                        cx = float(intrinsics.cx)
                        cy = float(intrinsics.cy)
                        x_cam = float((u_c - cx) * z_est / fx)
                        y_cam = float((v_c - cy) * z_est / fy)
                    else:
                        x_cam = float(det_xyz[i, 0].item())
                        y_cam = float(det_xyz[i, 1].item())

                    z_cam = float(z_est)
                    dist = float(np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2))
                else:
                    x_cam = float(det_xyz[i, 0].item())
                    y_cam = float(det_xyz[i, 1].item())
                    z_cam = float(det_xyz[i, 2].item())
                    dist = float(np.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2))
                    d_conf = float(score)
                    mask_source = "none"

                # 3D Physical Dimensions from model head (no heuristic multipliers)
                raw_l = float(det_lwh[i, 0].item()) if det_lwh.numel() > 0 else None
                raw_w = float(det_lwh[i, 1].item()) if det_lwh.numel() > 0 else None
                raw_h = float(det_lwh[i, 2].item()) if det_lwh.numel() > 0 else None
                # LWH must be positive; they come from exp-clamped head output
                dim_l = max(0.01, raw_l) if raw_l is not None else None
                dim_w = max(0.01, raw_w) if raw_w is not None else None
                dim_h = max(0.01, raw_h) if raw_h is not None else None
                yaw_deg = float(det_yaw[i, 0].item() * 180.0 / np.pi) if det_yaw.numel() > 0 else 0.0

                frame_objects.append({
                    "id": i + 1,
                    "class": label,
                    "score": score,
                    "bbox": [float(box[0].item()), float(box[1].item()), float(box[2].item()), float(box[3].item())],
                    "mask": obj_mask,
                    "mask_source": mask_source,
                    "x": x_cam,
                    "y": y_cam,
                    "z": z_cam,
                    "depth": z_cam,
                    "distance": dist,
                    "depth_confidence": float(d_conf),
                    "dimensions3D": {
                        "length": round(dim_l, 3) if dim_l is not None else None,
                        "width": round(dim_w, 3) if dim_w is not None else None,
                        "height": round(dim_h, 3) if dim_h is not None else None,
                    },
                    "yawDeg": round(yaw_deg, 1)
                })


            if tracker is not None:
                frame_objects = tracker.step(frame_objects, dt=dt)

            batch_objects.append(frame_objects)

        processed_outputs: Dict[str, Any] = {
            "detections": batch_detections,
            "objects": batch_objects
        }

        if "semantic_masks" in outputs:
            processed_outputs["semantic_probs"] = torch.softmax(outputs["semantic_masks"], dim=1)
            processed_outputs["semantic_labels"] = torch.argmax(outputs["semantic_masks"], dim=1)
        if "boundary_map" in outputs:
            processed_outputs["boundary_probs"] = torch.sigmoid(outputs["boundary_map"])
        if "instance_embeddings" in outputs:
            processed_outputs["instance_embeddings"] = outputs["instance_embeddings"]
        if "mask_quality" in outputs:
            processed_outputs["mask_quality_probs"] = torch.sigmoid(outputs["mask_quality"])
        if "depth_map" in outputs:
            processed_outputs["depth_map"] = outputs["depth_map"]
        if "terrain_elevation" in outputs:
            processed_outputs["terrain_elevation"] = outputs["terrain_elevation"]
        if "depth_inverse" in outputs:
            processed_outputs["depth_inverse"] = outputs["depth_inverse"]
        if "dense_xyz" in outputs:
            processed_outputs["dense_xyz"] = outputs["dense_xyz"]
        if "proto_masks" in outputs:
            processed_outputs["proto_masks"] = outputs["proto_masks"]

        return processed_outputs

def calculate_map_metrics(pred_boxes_list, pred_scores_list, pred_labels_list,
                          gt_boxes_list, gt_labels_list,
                          iou_thresholds=None,
                          cat_ids=None):
    """
    Exact, mathematically compliant COCO-standard mAP calculation using official pycocotools.
    Computes:
      - mAP50:95 (AP@[.50:.95], area=all)
      - mAP50 (AP@0.50, area=all)
      - mAP75 (AP@0.75, area=all)
      - APs (AP@[.50:.95], area < 32^2)
      - APm (AP@[.50:.95], 32^2 <= area < 96^2)
      - APl (AP@[.50:.95], area >= 96^2)
      - AR1, AR10, AR100 (Average Recall)
    Strictly forbids AP = prec * rec heuristics or arbitrary multipliers.
    """
    from pycocotools.coco import COCO  # type: ignore[import-untyped,import-not-found]
    from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped,import-not-found]

    # 1. Discover all categories
    if cat_ids is not None:
        all_categories = sorted(list(set(cat_ids)))
    else:
        found_cats = set()
        for g_lbl in gt_labels_list:
            if len(g_lbl) > 0:
                found_cats.update(g_lbl.squeeze(-1).tolist() if g_lbl.ndim > 1 else g_lbl.tolist())
        for p_lbl in pred_labels_list:
            if len(p_lbl) > 0:
                found_cats.update(p_lbl.squeeze(-1).tolist() if p_lbl.ndim > 1 else p_lbl.tolist())
        all_categories = sorted(list(found_cats)) if len(found_cats) > 0 else [0]

    coco_gt = COCO()
    images_info = []
    annotations_info = []
    categories_info = [{"id": int(c), "name": f"class_{c}", "supercategory": "object"} for c in all_categories]

    ann_id = 1
    num_images = len(gt_boxes_list)
    for img_id in range(num_images):
        images_info.append({
            "id": img_id,
            "height": 10000,
            "width": 10000
        })

        g_boxes = gt_boxes_list[img_id]
        g_labels = gt_labels_list[img_id]
        if len(g_boxes) == 0:
            continue

        if isinstance(g_boxes, torch.Tensor):
            g_boxes = g_boxes.cpu().numpy()
        if isinstance(g_labels, torch.Tensor):
            g_labels = g_labels.cpu().numpy()

        for box, lbl in zip(g_boxes, g_labels):
            cid = int(lbl[0] if hasattr(lbl, "__len__") and len(lbl) > 0 else lbl)
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h
            annotations_info.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt.dataset = {
        "images": images_info,
        "annotations": annotations_info,
        "categories": categories_info
    }
    coco_gt.createIndex()

    # 2. Build COCO detection predictions
    coco_dt_list = []
    for img_id in range(num_images):
        if img_id >= len(pred_boxes_list):
            continue
        p_boxes = pred_boxes_list[img_id]
        p_scores = pred_scores_list[img_id]
        p_labels = pred_labels_list[img_id]
        if len(p_boxes) == 0:
            continue

        if isinstance(p_boxes, torch.Tensor):
            p_boxes = p_boxes.cpu().numpy()
        if isinstance(p_scores, torch.Tensor):
            p_scores = p_scores.cpu().numpy()
        if isinstance(p_labels, torch.Tensor):
            p_labels = p_labels.cpu().numpy()

        for box, score, lbl in zip(p_boxes, p_scores, p_labels):
            cid = int(lbl[0] if hasattr(lbl, "__len__") and len(lbl) > 0 else lbl)
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            coco_dt_list.append({
                "image_id": img_id,
                "category_id": cid,
                "bbox": [x1, y1, w, h],
                "score": float(score)
            })

    empty_result = {
        "mAP50": 0.0,
        "mAP75": 0.0,
        "mAP50:95": 0.0,
        "APs": 0.0,
        "APm": 0.0,
        "APl": 0.0,
        "AR1": 0.0,
        "AR10": 0.0,
        "AR100": 0.0
    }

    if len(annotations_info) == 0 or len(coco_dt_list) == 0:
        return empty_result

    try:
        # pycocotools accepts a list of dicts at runtime, but stubs only specify `resFile: str`
        coco_dt = coco_gt.loadRes(cast(str, coco_dt_list))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if iou_thresholds is not None:
            coco_eval.params.iouThrs = np.array(iou_thresholds)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats

        def safe_stat(idx):
            val = float(stats[idx])
            return 0.0 if val < 0 else val

        return {
            "mAP50:95": safe_stat(0),
            "mAP50": safe_stat(1),
            "mAP75": safe_stat(2),
            "APs": safe_stat(3),
            "APm": safe_stat(4),
            "APl": safe_stat(5),
            "AR1": safe_stat(6),
            "AR10": safe_stat(7),
            "AR100": safe_stat(8)
        }
    except Exception as e:
        print(f"Warning: COCOeval encountered error: {e}")
        return empty_result

def calculate_miou(pred_sem: torch.Tensor, gt_sem: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
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
    Standard metric depth evaluation suite (Eigen / KITTI / NYU):
      - AbsRel: mean(|P - G| / G)
      - SqRel: mean((P - G)^2 / G)
      - RMSE: sqrt(mean((P - G)^2))
      - RMSE_log: sqrt(mean((log P - log G)^2))
      - delta1: mean(max(P/G, G/P) < 1.25)
      - delta2: mean(max(P/G, G/P) < 1.25^2)
      - delta3: mean(max(P/G, G/P) < 1.25^3)
    """
    v = valid_mask.bool()
    if not v.any():
        return {
            "AbsRel": 0.0,
            "SqRel": 0.0,
            "RMSE": 0.0,
            "RMSE_log": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0
        }

    p = pred_depth[v].clamp_min(1e-4)
    g = gt_depth[v].clamp_min(1e-4)

    abs_diff = torch.abs(p - g)
    abs_rel = torch.mean(abs_diff / g).item()
    sq_rel = torch.mean((abs_diff ** 2) / g).item()
    rmse = torch.sqrt(torch.mean((p - g) ** 2)).item()

    log_p = torch.log(p)
    log_g = torch.log(g)
    rmse_log = torch.sqrt(torch.mean((log_p - log_g) ** 2)).item()

    ratio = torch.max(p / g, g / p)
    delta1 = (ratio < 1.25).float().mean().item()
    delta2 = (ratio < (1.25 ** 2)).float().mean().item()
    delta3 = (ratio < (1.25 ** 3)).float().mean().item()

    return {
        "AbsRel": float(abs_rel),
        "SqRel": float(sq_rel),
        "RMSE": float(rmse),
        "RMSE_log": float(rmse_log),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3)
    }

def calculate_boundary_fscore(pred_bound: torch.Tensor, gt_bound: torch.Tensor, threshold: float = 0.5) -> float:
    p = (pred_bound > threshold).bool()
    g = (gt_bound > 0.5).bool()

    tp = (p & g).sum().item()
    fp = (p & ~g).sum().item()
    fn = (~p & g).sum().item()

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-6)
    return float(f1)

def calculate_bos_metrics(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> dict:
    """
    Evaluates 2D Bounding Box IoU and Box Overlap Score / Boundary Overlap Stability (BoS).
    Returns:
      - mean_iou: Mean Intersection over Union
      - mean_bos: Mean Box Overlap Score (joint IoU + center alignment + scale consistency)
      - bos_at_50: Percentage of detections with BoS >= 0.50
      - bos_at_75: Percentage of detections with BoS >= 0.75
    """
    from ..geometry.box_ops import box_iou_2d, calculate_box_overlap_score
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {"mean_iou": 0.0, "mean_bos": 0.0, "bos_at_50": 0.0, "bos_at_75": 0.0}

    ious = box_iou_2d(pred_boxes, gt_boxes) # (N, M)
    bos_mat = calculate_box_overlap_score(pred_boxes, gt_boxes) # (N, M)

    max_ious, _ = ious.max(dim=1)
    max_bos, _ = bos_mat.max(dim=1)

    mean_iou = float(max_ious.mean().item())
    mean_bos = float(max_bos.mean().item())
    bos_at_50 = float((max_bos >= 0.50).float().mean().item())
    bos_at_75 = float((max_bos >= 0.75).float().mean().item())

    return {
        "mean_iou": mean_iou,
        "mean_bos": mean_bos,
        "bos_at_50": bos_at_50,
        "bos_at_75": bos_at_75
    }

def calculate_3d_iou_and_bos(pred_xyz: torch.Tensor, pred_lwh: torch.Tensor, pred_yaw: torch.Tensor,
                             gt_xyz: torch.Tensor, gt_lwh: torch.Tensor, gt_yaw: torch.Tensor) -> dict:
    """
    Evaluates 3D Oriented Bounding Box IoU and 3D Boundary Overlap Score (BoS_3D).
    """
    from ..geometry.oriented_iou3d import oriented_iou_3d
    if len(pred_xyz) == 0 or len(gt_xyz) == 0:
        return {"mean_3d_iou": 0.0, "mean_3d_bos": 0.0, "iou3d_at_50": 0.0, "iou3d_at_75": 0.0}

    iou_3d = oriented_iou_3d(pred_xyz, pred_lwh, pred_yaw, gt_xyz, gt_lwh, gt_yaw)
    
    # 3D spatial alignment
    rho2_3d = ((pred_xyz - gt_xyz) ** 2).sum(dim=-1)
    min_c = torch.min(pred_xyz - 0.5 * pred_lwh, gt_xyz - 0.5 * gt_lwh)
    max_c = torch.max(pred_xyz + 0.5 * pred_lwh, gt_xyz + 0.5 * gt_lwh)
    diag2 = ((max_c - min_c) ** 2).sum(dim=-1).clamp_min(1e-4)

    bos_3d = (iou_3d * torch.exp(-torch.sqrt(rho2_3d / diag2))).clamp(0.0, 1.0)

    return {
        "mean_3d_iou": float(iou_3d.mean().item()),
        "mean_3d_bos": float(bos_3d.mean().item()),
        "iou3d_at_50": float((iou_3d >= 0.50).float().mean().item()),
        "iou3d_at_75": float((iou_3d >= 0.75).float().mean().item())
    }


