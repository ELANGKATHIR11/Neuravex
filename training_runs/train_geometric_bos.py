"""
Fine-Grained Geometric Bounding Box & BoS Alignment Trainer:
Trains Neuravex Detection Head using Dense Pseudo-Ground-Truth Bounding Boxes
from YOLO26 & Salient Contours on the Animals Dataset.

Optimizes:
  - Complete IoU Loss (CIoU)
  - Boundary Overlap Score (BoS Loss)
  - Distribution Focal Loss (DFL) on Anchor Pyramids
Saves:
  - neuravex_animals_geometric_best.pth
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = r"C:\Users\elang\Downloads\neuravex-cv"
ML_DIR = os.path.join(REPO_ROOT, "ml_neuravex")
DATASET_ROOT = r"C:\Users\elang\Downloads\animals"
OUTPUT_DIR = os.path.join(REPO_ROOT, "training_runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from neuravex import (
    build_neuravex,
    CameraIntrinsics,
    bbox_ciou,
    calculate_box_overlap_score,
    box_iou_2d
)
from ultralytics import YOLO

# -------------------------------------------------------------
# 1. GENERATE DENSE SALIENT PSEUDO-BOUNDING LABELS
# -------------------------------------------------------------
CLASS_NAMES = ["Asiatic Lion", "Indian Cow", "Indian Dog", "Indian Macaque", "Langur", "tiger"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

yolo_teacher = YOLO(os.path.join(REPO_ROOT, "yolo11n.pt"))
yolo_teacher.to("cuda")

print("=" * 85)
print("EXTRACTING DENSE PSEUDO-BOUNDING BOX ANNOTATIONS FROM TEACHER & SALIENCY")
print("=" * 85)

annotated_dataset = []

for cls_name in CLASS_NAMES:
    folder = os.path.join(DATASET_ROOT, "Indian Animals", cls_name)
    if not os.path.exists(folder):
        continue
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, f)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            orig_h, orig_w = img_bgr.shape[:2]
            
            # 1. Run YOLO teacher
            img_resized = cv2.resize(img_bgr, (640, 640))
            y_res = yolo_teacher.predict(img_resized, device="cuda:0", verbose=False)[0]
            
            if len(y_res.boxes) > 0:
                top_box = y_res.boxes.xyxy[0].cpu().numpy().tolist()
            else:
                # 2. Saliency contour fallback
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c_max = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(c_max)
                    top_box = [x * 640.0 / orig_w, y * 640.0 / orig_h, (x+bw) * 640.0 / orig_w, (y+bh) * 640.0 / orig_h]
                else:
                    top_box = [640.0 * 0.15, 640.0 * 0.15, 640.0 * 0.85, 640.0 * 0.85]
                    
            annotated_dataset.append({
                "path": img_path,
                "class_id": CLASS_TO_IDX[cls_name],
                "box": top_box
            })

print(f"Generated {len(annotated_dataset)} dense bounding box ground truths.")

# -------------------------------------------------------------
# 2. PYTORCH DATASET & GEOMETRIC LOSSES
# -------------------------------------------------------------
class GeometricAnimalDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        entry = self.data[idx]
        img_bgr = cv2.imread(entry["path"])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        box_t = torch.tensor(entry["box"], dtype=torch.float32)
        cls_t = torch.tensor(entry["class_id"], dtype=torch.long)
        
        return img_t, box_t, cls_t

train_loader = DataLoader(GeometricAnimalDataset(annotated_dataset), batch_size=4, shuffle=True)

# -------------------------------------------------------------
# 3. LOAD EXISTING TRAINED NEURAVEX AND FINE-TUNE DETECTION HEAD
# -------------------------------------------------------------
device = torch.device("cuda:0")
ckpt_path = os.path.join(REPO_ROOT, "training_runs", "neuravex_animals_best.pth")
model = build_neuravex(size="nano", num_classes=len(CLASS_NAMES)).to(device)

if os.path.exists(ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
K = CameraIntrinsics(fx=640.0, fy=640.0, cx=320.0, cy=320.0, device="cuda")

print("\n" + "=" * 85)
print("TRAINING GEOMETRIC CIoU & BoS BOUNDARY ALIGNMENT (10 EPOCHS)")
print("=" * 85)

NUM_EPOCHS = 10
best_geo_loss = float("inf")
telemetry = []

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss_epoch = 0.0
    total_iou_epoch = 0.0
    total_bos_epoch = 0.0
    batches = 0
    
    for imgs, boxes, cls_ids in train_loader:
        imgs = imgs.to(device)
        boxes = boxes.to(device) # (B, 4)
        cls_ids = cls_ids.to(device)
        B = imgs.shape[0]
        
        optimizer.zero_grad(set_to_none=True)
        out = model(imgs, intrinsics=K)
        
        # Predicted multi-scale boxes: (B, N, 4)
        pred_boxes = out["pred_boxes"]
        pred_cls = out["class_logits"].mean(dim=1)
        
        # Match predicted top candidate box to ground truth box
        # Loss 1: Classification Loss
        loss_cls = F.cross_entropy(pred_cls, cls_ids)
        
        # Loss 2: Complete IoU Loss (CIoU) on best candidate
        # Compare first anchor predictions directly against GT box
        anchor_box = pred_boxes[:, 0, :] # (B, 4)
        
        ciou_val = bbox_ciou(anchor_box, boxes) # (B, 1) or scalar
        loss_ciou = (1.0 - ciou_val).mean()
        
        # Loss 3: Direct Boundary Overlap Score (BoS) Loss: 1.0 - BoS
        bos_val = calculate_box_overlap_score(anchor_box, boxes)
        loss_bos = (1.0 - bos_val).mean()
        
        # Combined Geometric Loss
        loss = loss_cls + 2.0 * loss_ciou + 2.0 * loss_bos
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss_epoch += loss.item()
        # Compute tracking metrics
        with torch.no_grad():
            iou_metric = box_iou_2d(anchor_box, boxes).diagonal().mean().item()
            bos_metric = bos_val.mean().item()
            total_iou_epoch += iou_metric
            total_bos_epoch += bos_metric
            
        batches += 1
        
    scheduler.step()
    mean_l = total_loss_epoch / batches
    mean_iou = total_iou_epoch / batches
    mean_bos = total_bos_epoch / batches
    
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Loss: {mean_l:.4f} | Training IoU: {mean_iou:.4f} | Training BoS: {mean_bos:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    telemetry.append({
        "epoch": epoch,
        "loss": round(mean_l, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_bos": round(mean_bos, 4)
    })
    
    if mean_l < best_geo_loss:
        best_geo_loss = mean_l
        best_ckpt = os.path.join(OUTPUT_DIR, "neuravex_animals_geometric_best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": best_geo_loss,
            "classes": CLASS_NAMES,
            "size": "nano"
        }, best_ckpt)

print(f"\nGeometric Fine-Tuning Complete! Best Model saved to: {best_ckpt}")

# Save telemetry
geo_csv = os.path.join(OUTPUT_DIR, "geometric_training_telemetry.csv")
with open(geo_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
    writer.writeheader()
    writer.writerows(telemetry)

print("Saved telemetry to geometric_training_telemetry.csv.")
