"""
Evaluation script for YOLO27 v0.6 3D bounding boxes and DEM Depth on validation split.
"""
import torch
from torch.utils.data import DataLoader
from yolo27.models.yolo27 import build_yolo27
from yolo27.data.multitask_3d_dataset import Multitask3DDataset, multitask_3d_collate_fn
from yolo27.geometry.oriented_iou3d import oriented_iou_3d, rotated_rect_intersection_bev

def evaluate_3d_dem():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating YOLO27 3D + DEM model on {device}...")
    
    val_dataset = Multitask3DDataset(data_dir="data/synthetic_3d_dataset", split="val", img_size=320, num_classes=5)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=multitask_3d_collate_fn)
    
    model = build_yolo27(size="small", num_classes=5).to(device)
    ckpt = torch.load("weights/yolo27_v06_rtx5060_3d_dem.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_depth_rmse = 0.0
    total_depth_absrel = 0.0
    total_samples = 0

    xyz_errors = []
    lwh_errors = []
    yaw_errors = []
    bev_ious = []
    iou_3ds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            B = images.shape[0]
            total_samples += B
            
            outputs = model(images)
            pred_depth = outputs["depth_map"].squeeze() # (B, H, W)
            gt_depth = batch["depth"].to(device).squeeze() # (B, H, W)

            # DEM metrics
            mask = (gt_depth > 0.1) & (gt_depth < 60.0)
            if mask.sum() > 0:
                diff = pred_depth[mask] - gt_depth[mask]
                rmse = torch.sqrt(torch.mean(diff ** 2)).item()
                absrel = torch.mean(torch.abs(diff) / gt_depth[mask]).item()
                total_depth_rmse += rmse * B
                total_depth_absrel += absrel * B

            # 3D bounding box predictions (XYZ, LWH, Yaw)
            pred_xyz = outputs["pred_xyz"] # (B, N, 3)
            pred_lwh = outputs["pred_lwh"] # (B, N, 3)
            pred_sincos = outputs["pred_yaw_sincos"] # (B, N, 2)
            pred_yaw = torch.atan2(pred_sincos[..., 0:1], pred_sincos[..., 1:2]) # (B, N, 1)
            pred_scores = torch.sigmoid(outputs["class_logits"]) # (B, N, num_classes)
            max_scores, _ = pred_scores.max(dim=-1) # (B, N)

            for b in range(B):
                g_xyz = batch["gt_3d_xyz"][b].to(device) # (M, 3)
                g_lwh = batch["gt_3d_lwh"][b].to(device) # (M, 3)
                g_yaw = batch["gt_3d_yaw"][b].to(device) # (M, 1)
                mask_b = batch["mask_gt"][b].squeeze(-1) # (M,)
                
                g_xyz = g_xyz[mask_b]
                g_lwh = g_lwh[mask_b]
                g_yaw = g_yaw[mask_b]
                
                if len(g_xyz) == 0:
                    continue
                
                # Pick top predicted boxes with highest confidence
                topk_idx = max_scores[b].topk(k=min(10, max_scores.shape[1])).indices
                p_c = pred_xyz[b, topk_idx] # (K, 3)
                p_lwh = pred_lwh[b, topk_idx] # (K, 3)
                p_yaw = pred_yaw[b, topk_idx, 0] # (K,)

                # Evaluate pairwise oriented 3D IoU against ground truth
                for gt_idx in range(len(g_xyz)):
                    gt_c = g_xyz[gt_idx:gt_idx+1].expand_as(p_c)
                    gt_l = g_lwh[gt_idx:gt_idx+1].expand_as(p_lwh)
                    gt_y = g_yaw[gt_idx].expand_as(p_yaw)

                    iou_3d_vals = oriented_iou_3d(p_c, p_lwh, p_yaw, gt_c, gt_l, gt_y) # (K,)
                    
                    # BEV IoU
                    inter_bev = rotated_rect_intersection_bev(p_c, p_lwh, p_yaw, gt_c, gt_l, gt_y)
                    area_p = (p_lwh[:, 0] * p_lwh[:, 1]).clamp_min(1e-7)
                    area_g = (gt_l[:, 0] * gt_l[:, 1]).clamp_min(1e-7)
                    bev_iou_vals = inter_bev / (area_p + area_g - inter_bev + 1e-7)

                    best_k = iou_3d_vals.argmax()

                    xyz_err = torch.norm(p_c[best_k] - g_xyz[gt_idx]).item()
                    lwh_err = torch.norm(p_lwh[best_k] - g_lwh[gt_idx]).item()
                    yaw_err = torch.abs(torch.atan2(torch.sin(p_yaw[best_k] - g_yaw[gt_idx, 0]), 
                                                   torch.cos(p_yaw[best_k] - g_yaw[gt_idx, 0]))).item()

                    xyz_errors.append(xyz_err)
                    lwh_errors.append(lwh_err)
                    yaw_errors.append(yaw_err)
                    bev_ious.append(bev_iou_vals[best_k].item())
                    iou_3ds.append(iou_3d_vals[best_k].item())

    mean_rmse = total_depth_rmse / max(1, total_samples)
    mean_absrel = total_depth_absrel / max(1, total_samples)
    mean_xyz_err = sum(xyz_errors) / max(1, len(xyz_errors))
    mean_lwh_err = sum(lwh_errors) / max(1, len(lwh_errors))
    mean_yaw_err_deg = (sum(yaw_errors) / max(1, len(yaw_errors))) * (180.0 / 3.14159265)
    mean_bev_iou = sum(bev_ious) / max(1, len(bev_ious))
    mean_3d_iou = sum(iou_3ds) / max(1, len(iou_3ds))

    print("\n" + "="*70)
    print(" YOLO27 v0.6 3D & DEM VALIDATION BENCHMARK RESULTS")
    print("="*70)
    print(f"Validation Samples Evaluated : {total_samples}")
    print(f"Total 3D GT Objects Evaluated: {len(xyz_errors)}")
    print(f"DEM Depth RMSE               : {mean_rmse:.4f} meters")
    print(f"DEM Depth AbsRel             : {mean_absrel * 100:.2f}%")
    print(f"3D Center Error (XYZ)        : {mean_xyz_err:.4f} meters")
    print(f"3D Dimension Error (LWH)     : {mean_lwh_err:.4f} meters")
    print(f"3D Yaw Orientation Error     : {mean_yaw_err_deg:.2f}°")
    print(f"Mean BEV IoU                 : {mean_bev_iou * 100:.2f}%")
    print(f"Mean Oriented 3D IoU         : {mean_3d_iou * 100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    evaluate_3d_dem()
