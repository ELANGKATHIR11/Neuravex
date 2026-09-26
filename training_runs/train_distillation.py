"""
Production Knowledge Distillation & Training Script:
Trains Neuravex Foundation Model using YOLO26 (YOLO11) as teacher on the Animals Dataset.
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU.
Saves:
  - Best distilled student weights checkpoint (neuravex_animals_best.pth)
  - Training telemetry & loss curves
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

from neuravex import build_neuravex, CameraIntrinsics
from ultralytics import YOLO

# -------------------------------------------------------------
# 1. DATASET WITH MULTI-MODAL PRE-PROCESSING
# -------------------------------------------------------------
CLASS_NAMES = ["Asiatic Lion", "Indian Cow", "Indian Dog", "Indian Macaque", "Langur", "tiger"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

class AnimalDataset(Dataset):
    def __init__(self, root_dir, img_size=640, augment=True):
        self.img_size = img_size
        self.augment = augment
        self.entries = []
        
        for cls_name in CLASS_NAMES:
            cls_folder = os.path.join(root_dir, "Indian Animals", cls_name)
            if not os.path.exists(cls_folder):
                continue
            for f in sorted(os.listdir(cls_folder)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.entries.append({
                        "path": os.path.join(cls_folder, f),
                        "class_name": cls_name,
                        "class_id": CLASS_TO_IDX[cls_name]
                    })
                    
    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_bgr = cv2.imread(entry["path"])
        if img_bgr is None:
            # Fallback black image if corrupted
            img_bgr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Resize to fixed square resolution
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        
        # Horizontal flip augmentation
        if self.augment and np.random.rand() > 0.5:
            img_resized = cv2.flip(img_resized, 1)
            
        tensor_img = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        
        return {
            "image": tensor_img,
            "class_id": torch.tensor(entry["class_id"], dtype=torch.long),
            "class_name": entry["class_name"]
        }

# -------------------------------------------------------------
# 2. TRAINING SETUP & DISTILLATION PIPELINE
# -------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"TRAINING NEURAVEX WITH YOLO26 TEACHER ON RTX 5060 ({device})")
print("=" * 80)

# Instantiate Teacher: YOLO11n (YOLO26 baseline)
yolo_teacher_path = os.path.join(REPO_ROOT, "yolo11n.pt")
print(f"Loading Teacher Model from: {yolo_teacher_path}")
teacher_yolo = YOLO(yolo_teacher_path)
teacher_yolo.to(device)

# Instantiate Student: Neuravex (Nano tier for ultra-efficiency & edge readiness)
print("Building Neuravex Student Model (Tier: Nano, 2.33M Params)...")
student_model = build_neuravex(size="nano", num_classes=len(CLASS_NAMES)).to(device)
student_model.train()

# Dataset and DataLoader
dataset = AnimalDataset(DATASET_ROOT, img_size=640, augment=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
print(f"Dataset loaded: {len(dataset)} training images across {len(CLASS_NAMES)} classes.")

# Optimizer & LR Scheduler
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
K = CameraIntrinsics(fx=640.0, fy=640.0, cx=320.0, cy=320.0, device="cuda")

# -------------------------------------------------------------
# 3. PRODUCTION DISTILLATION LOOP
# -------------------------------------------------------------
NUM_EPOCHS = 10
best_loss = float("inf")
epoch_losses = []
telemetry_records = []

start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_total_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_kd_loss = 0.0
    epoch_depth_loss = 0.0
    batch_count = 0
    
    student_model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        imgs = batch["image"].to(device)
        class_ids = batch["class_id"].to(device)
        B = imgs.shape[0]
        
        # 1. Teacher Forward Pass (YOLO26 Zero-shot feature & box priors)
        with torch.no_grad():
            # YOLO expects BGR numpy or uint8 tensor
            imgs_uint8 = (imgs * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            teacher_preds = teacher_yolo.predict(list(imgs_uint8), device="cuda:0", verbose=False)
            
            # Extract teacher pseudo-bounding priors
            teacher_confidence = []
            for pred in teacher_preds:
                if len(pred.boxes) > 0:
                    top_conf = pred.boxes.conf[0].item()
                else:
                    top_conf = 0.5
                teacher_confidence.append(top_conf)
            t_conf_tensor = torch.tensor(teacher_confidence, device=device).unsqueeze(-1)
            
        # 2. Student Forward Pass (Neuravex multi-task)
        optimizer.zero_grad(set_to_none=True)
        student_out = student_model(imgs, intrinsics=K)
        
        # 3. Multi-Task Losses:
        # A. Classification Loss on animal category
        student_cls_logits = student_out["class_logits"] # (B, N, num_classes)
        # Global max-pooled representation across scales
        pooled_cls = student_cls_logits.mean(dim=1) # (B, num_classes)
        loss_cls = F.cross_entropy(pooled_cls, class_ids)
        
        # B. Teacher Distillation (Knowledge alignment with YOLO confidence)
        loss_kd = F.mse_loss(torch.sigmoid(pooled_cls.max(dim=-1, keepdim=True)[0]), t_conf_tensor)
        
        # C. Self-Supervised Depth Regularization (CameraAwareDEM smooth surface prior)
        depth_map = student_out.get("depth_map", None)
        if depth_map is not None:
            # 2nd order smoothness gradient penalty
            dx = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
            dy = torch.abs(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :])
            loss_depth = (dx.mean() + dy.mean()) * 0.1
        else:
            loss_depth = torch.tensor(0.0, device=device)
            
        total_loss = loss_cls + 0.5 * loss_kd + loss_depth
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=5.0)
        optimizer.step()
        
        epoch_total_loss += total_loss.item()
        epoch_cls_loss += loss_cls.item()
        epoch_kd_loss += loss_kd.item()
        epoch_depth_loss += loss_depth.item()
        batch_count += 1
        
    scheduler.step()
    
    avg_loss = epoch_total_loss / batch_count
    avg_cls = epoch_cls_loss / batch_count
    avg_kd = epoch_kd_loss / batch_count
    epoch_losses.append(avg_loss)
    
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | Total Loss: {avg_loss:.4f} | Cls Loss: {avg_cls:.4f} | KD Alignment: {avg_kd:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    telemetry_records.append({
        "epoch": epoch,
        "total_loss": round(avg_loss, 4),
        "cls_loss": round(avg_cls, 4),
        "kd_loss": round(avg_kd, 4),
        "lr": scheduler.get_last_lr()[0]
    })
    
    # Save Best Weights Checkpoint
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(OUTPUT_DIR, "neuravex_animals_best.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": student_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": best_loss,
            "classes": CLASS_NAMES,
            "size": "nano"
        }, checkpoint_path)

elapsed_sec = time.time() - start_time
print(f"\nTraining Complete in {elapsed_sec:.1f}s! Best Model Checkpoint saved to: {checkpoint_path}")

# -------------------------------------------------------------
# 4. EXPORT TELEMETRY AND CONVERGENCE CURVE
# -------------------------------------------------------------
# Save CSV
telemetry_csv = os.path.join(OUTPUT_DIR, "training_telemetry.csv")
with open(telemetry_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(telemetry_records[0].keys()))
    writer.writeheader()
    writer.writerows(telemetry_records)

# Plot Loss Curve
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
epochs_x = [t["epoch"] for t in telemetry_records]
losses_y = [t["total_loss"] for t in telemetry_records]
cls_y = [t["cls_loss"] for t in telemetry_records]
kd_y = [t["kd_loss"] for t in telemetry_records]

ax.plot(epochs_x, losses_y, marker='o', linewidth=2.5, color='#38bdf8', label='Total Distilled Loss')
ax.plot(epochs_x, cls_y, marker='s', linewidth=2, color='#34d399', label='Animal Classification Loss')
ax.plot(epochs_x, kd_y, marker='^', linewidth=1.5, color='#fbbf24', label='YOLO26 KD Alignment Loss')
ax.set_title("Neuravex Knowledge Distillation Training Loss Curve", fontsize=13, weight='bold')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "training_loss_curve.png")
fig.savefig(curve_path)
plt.close(fig)

print(f"Convergence graph saved to: {curve_path}")
