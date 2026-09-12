import time
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo27.models.yolo27 import build_yolo27
from yolo27.engine.evaluator import (
    YOLO27InferencePostProcessor,
    calculate_map_metrics,
    calculate_miou,
    calculate_boundary_fscore,
    calculate_depth_metrics
)
from yolo27.data.real_vegetable_dataset import RealVegetableDataset, real_dataset_collate_fn

def evaluate_on_dgpu():
    print("=" * 75)
    print("   EVALUATING YOLO27 v0.6 ON REAL VALIDATION DATASET USING RTX 5060")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = "weights/yolo27_v06_rtx5060_vegetables.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}!"

    # 1. Load trained checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_yolo27(size="small", num_classes=4).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded trained checkpoint from Epoch {checkpoint['epoch']} on {checkpoint['device']}.")

    # 2. Validation DataLoader
    val_dataset = RealVegetableDataset(root_dir="F:/Vegetable-Object-Detection", split="valid", img_size=320, num_classes=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=real_dataset_collate_fn)
    print(f"Validation images: {len(val_dataset)}.\n")

    post_processor = YOLO27InferencePostProcessor(conf_thresh=0.15, iou_thresh=0.45)

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    sem_ious = []
    boundary_f1s = []
    depth_rmses = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            gt_boxes = batch["gt_boxes"].to(device)
            gt_labels = batch["gt_labels"].to(device)
            mask_gt = batch["mask_gt"].to(device)
            sem_gt = batch["semantic_masks"].to(device)
            bound_gt = batch["boundary_maps"].to(device)
            depth_gt = batch["depth"].to(device)
            valid_depth = batch["valid_depth"].to(device)

            out = model(images)
            processed = post_processor(out)

            # Detection metrics aggregation
            for b in range(images.shape[0]):
                det = processed["detections"][b]
                all_pred_boxes.append(det["boxes"].cpu())
                all_pred_scores.append(det["scores"].cpu())
                all_pred_labels.append(det["labels"].cpu())

                valid_g = mask_gt[b].squeeze(-1)
                all_gt_boxes.append(gt_boxes[b, valid_g].cpu())
                all_gt_labels.append(gt_labels[b, valid_g].squeeze(-1).cpu())

            # Semantic mIoU
            pred_sem = processed["semantic_labels"]
            miou_b = calculate_miou(pred_sem.cpu(), sem_gt.cpu(), num_classes=4)
            sem_ious.append(miou_b)

            # Boundary F1
            pred_bound = processed["boundary_probs"]
            f1_b = calculate_boundary_fscore(pred_bound.cpu(), bound_gt.cpu())
            boundary_f1s.append(f1_b)

            # Depth RMSE
            pred_d = processed["depth_map"]
            d_metric = calculate_depth_metrics(pred_d.cpu(), depth_gt.cpu(), valid_depth.cpu())
            depth_rmses.append(d_metric["RMSE"])

    # Final metrics
    det_metrics = calculate_map_metrics(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels
    )
    mean_miou = sum(sem_ious) / max(len(sem_ious), 1)
    mean_f1 = sum(boundary_f1s) / max(len(boundary_f1s), 1)
    mean_rmse = sum(depth_rmses) / max(len(depth_rmses), 1)

    print("=" * 75)
    print("REAL VALIDATION RESULTS (F:\\Vegetable-Object-Detection\\valid):")
    print(f"  * 2D Detection mAP50:     {det_metrics['mAP50'] * 100:.2f} %")
    print(f"  * 2D Detection mAP50:95:  {det_metrics['mAP50:95'] * 100:.2f} %")
    print(f"  * Semantic Segmentation: {mean_miou * 100:.2f} % mIoU")
    print(f"  * Boundary Detection:    {mean_f1 * 100:.2f} % F1-score")
    print(f"  * Depth Estimation:      {mean_rmse:.4f} m RMSE")
    print("=" * 75)

if __name__ == "__main__":
    evaluate_on_dgpu()
