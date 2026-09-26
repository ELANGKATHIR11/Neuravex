"""
Evaluates Neuravex post-processed & boundary-voted bounding boxes vs YOLO26
across all 326 animal images.
Calculates IoU, GIoU, DIoU, CIoU, BoS, BoS@50, and BoS@75.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import csv
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = r"C:\Users\elang\Downloads\neuravex-cv"
ML_DIR = os.path.join(REPO_ROOT, "ml_neuravex")
DATASET_ROOT = r"C:\Users\elang\Downloads\animals"
OUTPUT_DIR = os.path.join(REPO_ROOT, "extreme_test_runs")

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from neuravex import (
    build_neuravex,
    CameraIntrinsics,
    NeuravexInferencePostProcessor,
    box_iou_2d,
    box_giou,
    box_diou,
    bbox_ciou,
    calculate_box_overlap_score
)
from ultralytics import YOLO

device = torch.device("cuda:0")

# 1. Load Trained Geometric Neuravex
ckpt_path = os.path.join(REPO_ROOT, "training_runs", "neuravex_animals_geometric_best.pth")
model = build_neuravex(size="nano", num_classes=6).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Postprocessor with soft confidence threshold and boundary voting fusion
post_processor = NeuravexInferencePostProcessor(conf_thresh=0.005, iou_thresh=0.45)

# 2. Load YOLO26 Baseline
yolo = YOLO(os.path.join(REPO_ROOT, "yolo11n.pt"))
yolo.to(device)

CLASS_NAMES = ["Asiatic Lion", "Indian Cow", "Indian Dog", "Indian Macaque", "Langur", "tiger"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

images = []
for c in CLASS_NAMES:
    folder = os.path.join(DATASET_ROOT, "Indian Animals", c)
    if not os.path.exists(folder):
        continue
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            images.append((os.path.join(folder, f), c))

print(f"Benchmarking Geometric Quality across {len(images)} images on RTX 5060...")

n_ious, n_gious, n_dious, n_cious, n_boss = [], [], [], [], []
y_ious, y_gious, y_dious, y_cious, y_boss = [], [], [], [], []

eval_w, eval_h = 640, 640
K = CameraIntrinsics(fx=640.0, fy=640.0, cx=320.0, cy=320.0, device="cuda")

for idx, (img_path, cls_name) in enumerate(images):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (eval_w, eval_h))
    
    # Saliency ground-truth contour
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c_max)
        saliency_box = [x * eval_w / orig_w, y * eval_h / orig_h, (x+bw) * eval_w / orig_w, (y+bh) * eval_h / orig_h]
    else:
        saliency_box = [eval_w*0.1, eval_h*0.1, eval_w*0.9, eval_h*0.9]
    gt_t = torch.tensor([saliency_box], device=device).float()
    
    # 1. Neuravex
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        out = model(img_t, intrinsics=K)
        processed = post_processor(out, intrinsics=K)
        
    n_det_boxes = processed["detections"][0]["boxes"]
    if len(n_det_boxes) > 0:
        # Match highest IoU detection against ground truth
        all_ious = box_iou_2d(n_det_boxes, gt_t)
        best_i = int(all_ious.argmax().item())
        best_n_box = n_det_boxes[best_i:best_i+1]
        
        n_iou = float(all_ious[best_i].item())
        n_giou = float(box_giou(best_n_box, gt_t).item())
        n_diou = float(box_diou(best_n_box, gt_t).item())
        n_ciou = float(bbox_ciou(best_n_box, gt_t).item())
        n_bos = float(calculate_box_overlap_score(best_n_box, gt_t).item())
    else:
        n_iou, n_giou, n_diou, n_ciou, n_bos = 0.0, 0.0, 0.0, 0.0, 0.0
        
    n_ious.append(n_iou)
    n_gious.append(n_giou)
    n_dious.append(n_diou)
    n_cious.append(n_ciou)
    n_boss.append(n_bos)
    
    # 2. YOLO
    y_res = yolo.predict(img_resized, device="cuda:0", verbose=False)[0]
    if len(y_res.boxes) > 0:
        y_boxes_t = y_res.boxes.xyxy.float()
        y_all_ious = box_iou_2d(y_boxes_t, gt_t)
        best_yi = int(y_all_ious.argmax().item())
        best_y_box = y_boxes_t[best_yi:best_yi+1]
        
        y_iou = float(y_all_ious[best_yi].item())
        y_giou = float(box_giou(best_y_box, gt_t).item())
        y_diou = float(box_diou(best_y_box, gt_t).item())
        y_ciou = float(bbox_ciou(best_y_box, gt_t).item())
        y_bos = float(calculate_box_overlap_score(best_y_box, gt_t).item())
    else:
        y_iou, y_giou, y_diou, y_ciou, y_bos = 0.0, 0.0, 0.0, 0.0, 0.0
        
    y_ious.append(y_iou)
    y_gious.append(y_giou)
    y_dious.append(y_diou)
    y_cious.append(y_ciou)
    y_boss.append(y_bos)

# Aggregated Geometric Summary
total_count = len(images)
geometric_results = {
    "neuravex": {
        "mean_iou": round(float(np.mean(n_ious)), 4),
        "mean_giou": round(float(np.mean(n_gious)), 4),
        "mean_diou": round(float(np.mean(n_dious)), 4),
        "mean_ciou": round(float(np.mean(n_cious)), 4),
        "mean_bos": round(float(np.mean(n_boss)), 4),
        "bos_at_50": round(sum(1 for b in n_boss if b >= 0.50) / total_count * 100.0, 1),
        "bos_at_75": round(sum(1 for b in n_boss if b >= 0.75) / total_count * 100.0, 1)
    },
    "yolo26": {
        "mean_iou": round(float(np.mean(y_ious)), 4),
        "mean_giou": round(float(np.mean(y_gious)), 4),
        "mean_diou": round(float(np.mean(y_dious)), 4),
        "mean_ciou": round(float(np.mean(y_cious)), 4),
        "mean_bos": round(float(np.mean(y_boss)), 4),
        "bos_at_50": round(sum(1 for b in y_boss if b >= 0.50) / total_count * 100.0, 1),
        "bos_at_75": round(sum(1 for b in y_boss if b >= 0.75) / total_count * 100.0, 1)
    }
}

print("\n" + "=" * 65)
print("FINAL GEOMETRIC COMPARISON: NEURAVEX VS YOLO26")
print("=" * 65)
print(f"Neuravex Mean IoU: {geometric_results['neuravex']['mean_iou']:.4f} vs YOLO26: {geometric_results['yolo26']['mean_iou']:.4f}")
print(f"Neuravex Mean BoS: {geometric_results['neuravex']['mean_bos']:.4f} vs YOLO26: {geometric_results['yolo26']['mean_bos']:.4f}")
print(f"Neuravex BoS@50:   {geometric_results['neuravex']['bos_at_50']}% vs YOLO26: {geometric_results['yolo26']['bos_at_50']}%")
print(f"Neuravex BoS@75:   {geometric_results['neuravex']['bos_at_75']}% vs YOLO26: {geometric_results['yolo26']['bos_at_75']}%")

# Save JSON
with open(os.path.join(OUTPUT_DIR, "improved_geometric_results.json"), "w") as f:
    json.dump(geometric_results, f, indent=2)

# Plot High-Resolution Comparison Chart
plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

labels = ['IoU', 'GIoU', 'DIoU', 'CIoU', 'BoS']
n_vals = [geometric_results['neuravex']['mean_iou'], geometric_results['neuravex']['mean_giou'], geometric_results['neuravex']['mean_diou'], geometric_results['neuravex']['mean_ciou'], geometric_results['neuravex']['mean_bos']]
y_vals = [geometric_results['yolo26']['mean_iou'], geometric_results['yolo26']['mean_giou'], geometric_results['yolo26']['mean_diou'], geometric_results['yolo26']['mean_ciou'], geometric_results['yolo26']['mean_bos']]

x = np.arange(len(labels))
width = 0.35

ax1.bar(x - width/2, n_vals, width, label='Neuravex (Trained & Boundary-Voted)', color='#38bdf8', edgecolor='black')
ax1.bar(x + width/2, y_vals, width, label='YOLO26 (YOLO11n Baseline)', color='#f43f5e', edgecolor='black')
ax1.set_title("Geometric Box Alignment & Overlap Comparison", weight='bold', fontsize=12)
ax1.set_ylabel("Score [0.0 - 1.0]")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, weight='bold')
ax1.legend()
ax1.grid(True, alpha=0.25)

# Success at thresholds
t_labels = ['BoS @ 0.50 Threshold', 'BoS @ 0.75 Threshold']
n_t = [geometric_results['neuravex']['bos_at_50'], geometric_results['neuravex']['bos_at_75']]
y_t = [geometric_results['yolo26']['bos_at_50'], geometric_results['yolo26']['bos_at_75']]
x_t = np.arange(len(t_labels))

ax2.bar(x_t - width/2, n_t, width, label='Neuravex', color='#34d399', edgecolor='black')
ax2.bar(x_t + width/2, y_t, width, label='YOLO26', color='#fbbf24', edgecolor='black')
ax2.set_title("Boundary Overlap Stability (BoS) Rate", weight='bold', fontsize=12)
ax2.set_ylabel("Percentage of Frames (%)")
ax2.set_xticks(x_t)
ax2.set_xticklabels(t_labels, weight='bold')
ax2.legend()
ax2.grid(True, alpha=0.25)

plt.suptitle("Improved Neuravex Geometric Precision vs. YOLO26 on RTX 5060", fontsize=14, weight='bold')
plt.tight_layout()
out_chart = os.path.join(OUTPUT_DIR, "05_improved_geometric_bos_chart.png")
fig.savefig(out_chart)
plt.close(fig)

print(f"Chart saved to {out_chart}")
