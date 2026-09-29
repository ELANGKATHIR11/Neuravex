import os
import sys
import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

REPO_ROOT = Path("c:/Users/elang/Downloads/neuravex-cv")
ML_DIR = REPO_ROOT / "ml_neuravex"
sys.path.insert(0, str(ML_DIR))

from neuravex.models.neuravex import build_neuravex
from ultralytics import YOLO

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

manifest_path = ML_DIR / "publishable_benchmark" / "dataset_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

classes = manifest["classes"]
train_records = [r for r in manifest["images"] if r["split"] == "train"]
val_records = [r for r in manifest["images"] if r["split"] == "val"]
test_records = [r for r in manifest["images"] if r["split"] == "test"]

# Load Teacher
teacher_path = ML_DIR / "publishable_benchmark" / "weights" / "yolo_train_YOLO11m-cls" / "run" / "weights" / "best.pt"
teacher = YOLO(str(teacher_path)).model.to(device).eval()

hooked = {}
def hook_pool(m, i, o):
    hooked["pool"] = o
teacher.model[10].pool.register_forward_hook(hook_pool)

class CleanAnimalsDataset(Dataset):
    def __init__(self, records, is_train=True):
        self.is_train = is_train
        self.samples = []
        for r in records:
            img = cv2.imread(r["split_path"])
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            # Store as uint8 [3, 224, 224]
            t = torch.from_numpy(img).permute(2, 0, 1)
            self.samples.append((t, r["class_id"], r["split_path"], r["class_name"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, label, path, cname = self.samples[idx]
        img = t.float() / 255.0  # EXACTLY MATCHES YOLOV11 PREPROCESSING!
        if self.is_train:
            # Subtle random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
        return img, label

train_ds = CleanAnimalsDataset(train_records, is_train=True)
val_ds = CleanAnimalsDataset(val_records, is_train=False)
test_ds = CleanAnimalsDataset(test_records, is_train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

class SOTANeuravexV2(nn.Module):
    def __init__(self, variant="edge", num_classes=6, teacher_dim=1280):
        super().__init__()
        self.neuravex = build_neuravex(variant, num_classes=num_classes)
        base_c = self.neuravex.base_c
        neck_c = base_c * 4
        in_dim = neck_c * 3  # q3 + q4 + q5
        p4_c = base_c * 8
        
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_dim // 4, in_dim, bias=False),
            nn.Sigmoid()
        )
        
        # Penultimate projection head matching 1280-dim teacher manifold
        self.teacher_proj = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, teacher_dim),
            nn.GELU()
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Dynamic Early Exit on P4
        self.early_exit_head = nn.Sequential(
            nn.BatchNorm1d(p4_c),
            nn.Linear(p4_c, num_classes)
        )

    def forward(self, x, early_exit_threshold=None):
        p3, p4, p5 = self.neuravex.backbone(x)
        f_early = p4.mean(dim=[-2, -1])
        logits_early = self.early_exit_head(f_early)
        
        if early_exit_threshold is not None and x.size(0) == 1:
            conf, _ = torch.softmax(logits_early, dim=-1).max(dim=-1)
            if conf.item() >= early_exit_threshold:
                dummy_proj = torch.zeros(1, 1280, device=x.device)
                return logits_early, dummy_proj, logits_early
                
        q3, q4, q5 = self.neuravex.neck(p3, p4, p5)
        f3 = q3.mean(dim=[-2, -1])
        f4 = q4.mean(dim=[-2, -1])
        f5 = q5.mean(dim=[-2, -1])
        f_multi = torch.cat([f3, f4, f5], dim=1)
        f_calibrated = f_multi * self.channel_gate(f_multi)
        
        proj_1280 = self.teacher_proj(f_calibrated)
        logits_full = self.classifier(f_calibrated)
        return logits_full, proj_1280, logits_early

    def switch_to_deploy(self):
        self.neuravex.switch_to_deploy()

def evaluate(model, loader, threshold=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits, _, logits_early = model(imgs, early_exit_threshold=threshold)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    return acc, f1, y_true, y_pred

# Run training
variant = "edge"
model = SOTANeuravexV2(variant, num_classes=6, teacher_dim=1280).to(device)

epochs = 80
optimizer = optim.AdamW([
    {"params": model.neuravex.parameters(), "lr": 4e-4, "weight_decay": 1e-4},
    {"params": model.channel_gate.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.teacher_proj.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1.5e-3, "weight_decay": 1e-4},
    {"params": model.early_exit_head.parameters(), "lr": 1.5e-3, "weight_decay": 1e-4}
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

print(f"--- Training SOTA Neuravex-{variant.upper()} with 1280-Dim Feature Manifold Alignment ---")
best_val_acc = 0.0
best_test_metrics = None

for epoch in range(1, epochs + 1):
    model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            t_logits = teacher(imgs)
            if isinstance(t_logits, tuple): t_logits = t_logits[0]
            t_feat = hooked["pool"].squeeze(-1).squeeze(-1)  # [B, 1280]
            
        optimizer.zero_grad()
        s_logits, s_proj, s_early = model(imgs)
        
        loss_ce = F.cross_entropy(s_logits, labels, label_smoothing=0.05)
        loss_early = F.cross_entropy(s_early, labels, label_smoothing=0.05)
        
        # Soft logit distillation (T=2.0)
        T = 2.0
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T * T)
        
        # 1280-Dim True Semantic Feature Manifold Alignment
        cos_sim = F.cosine_similarity(s_proj, t_feat, dim=-1)
        loss_feat = (1.0 - cos_sim).mean()
        
        loss = loss_ce + 0.3 * loss_early + 0.8 * loss_kd + 0.1 * loss_feat
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        
    scheduler.step()
    
    val_acc, val_f1, _, _ = evaluate(model, val_loader)
    
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        t_acc, t_f1, _, _ = evaluate(model, test_loader)
        best_test_metrics = (t_acc, t_f1)
        print(f"Epoch {epoch:2d}/{epochs:2d}: Val Acc={val_acc*100:5.1f}% (F1={val_f1:.4f}) | Test Acc={t_acc*100:5.1f}% (Macro-F1={t_f1:.4f}) [BEST!]")
    elif epoch % 10 == 0:
        print(f"Epoch {epoch:2d}/{epochs:2d}: Val Acc={val_acc*100:5.1f}% (F1={val_f1:.4f})")

print(f"\nFinal Peak Results: Val Acc={best_val_acc*100:.2f}%, Test Acc={best_test_metrics[0]*100:.2f}%, Test Macro-F1={best_test_metrics[1]:.4f}")
