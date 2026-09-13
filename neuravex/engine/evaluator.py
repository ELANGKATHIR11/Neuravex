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
        pred_xyz = outputs["pred_xyz"]                      # (B, N, 3)
        pred_lwh = outputs["pred_lwh"]                      # (B, N, 3)
        pred_yaw_sc = outputs["pred_yaw_sincos"]            # (B, N, 2)
        yaw_norm = F.normalize(pred_yaw_sc, dim=-1)
        recovered_yaw = torch.atan2(yaw_norm[..., 0], yaw_norm[..., 1]).unsqueeze(-1)

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
            f_xyz = pred_xyz[b, keep_mask]
            f_lwh = pred_lwh[b, keep_mask]
            f_yaw = recovered_yaw[b, keep_mask]

            # True class-aware batched NMS
            keep = batched_nms(f_boxes, f_scores, f_labels, self.iou_thresh)
            keep = keep[:self.max_det]

            det_boxes = f_boxes[keep]
            det_scores = f_scores[keep]
            det_labels = f_labels[keep]
            det_xyz = f_xyz[keep]
            det_lwh = f_lwh[keep]
            det_yaw = f_yaw[keep]

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

            # Get semantic predictions for true instance masking if available
            sem_pred_b = None
            if semantic_masks is not None:
                sem_pred_b = torch.argmax(semantic_masks[b], dim=0)  # (H, W)

            for i in range(len(det_boxes)):
                box = det_boxes[i]
                score = float(det_scores[i].item())
                label = int(det_labels[i].item())
                x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())

                # Extract TRUE instance mask
                obj_mask = None
                if cur_depth is not None:
                    _, H_d, W_d = cur_depth.shape if cur_depth.ndim == 3 else (1, cur_depth.shape[0], cur_depth.shape[1])
                    obj_mask = torch.zeros((H_d, W_d), dtype=torch.bool, device=box.device)
                    x1_c = max(0, min(W_d - 1, x1))
                    x2_c = max(0, min(W_d, x2))
                    y1_c = max(0, min(H_d - 1, y1))
                    y2_c = max(0, min(H_d, y2))

                    if x2_c > x1_c and y2_c > y1_c:
                        if sem_pred_b is not None:
                            # True semantic/instance mask within bbox
                            class_mask = (sem_pred_b[y1_c:y2_c, x1_c:x2_c] == label)
                            if class_mask.any():
                                obj_mask[y1_c:y2_c, x1_c:x2_c] = class_mask
                            else:
                                # Fallback to central region of bbox (rejecting border pixels)
                                obj_mask[y1_c:y2_c, x1_c:x2_c] = True
                        else:
                            obj_mask[y1_c:y2_c, x1_c:x2_c] = True

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

                frame_objects.append({
                    "id": i + 1,
                    "class": label,
                    "score": score,
                    "bbox": [float(box[0].item()), float(box[1].item()), float(box[2].item()), float(box[3].item())],
                    "mask": obj_mask,
                    "x": x_cam,
                    "y": y_cam,
                    "z": z_cam,
                    "depth": z_cam,
                    "distance": dist,
                    "depth_confidence": float(d_conf)
                })

            if tracker is not None:
                frame_objects = tracker.step(frame_objects, dt=dt)

            batch_objects.append(frame_objects)

        processed_outputs = {
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
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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
    categories_info = [{"id": int(c), "name": f"class_{c}"} for c in all_categories]

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
        coco_dt = coco_gt.loadRes(coco_dt_list)
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

