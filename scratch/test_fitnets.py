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

# Teacher model
teacher_path = ML_DIR / "publishable_benchmark" / "weights" / "yolo_train_YOLO11m-cls" / "run" / "weights" / "best.pt"
teacher = YOLO(str(teacher_path)).model.to(device).eval()

teacher_feats = {}
def hook_fn(name):
    def hook(m, inp, outp):
        teacher_feats[name] = outp
    return hook

teacher.model[4].register_forward_hook(hook_fn("p3"))
teacher.model[6].register_forward_hook(hook_fn("p4"))
teacher.model[9].register_forward_hook(hook_fn("p5"))

class PreloadedDataset(Dataset):
    def __init__(self, records, is_train=True):
        self.is_train = is_train
        self.data = []
        for r in records:
            img = cv2.imread(r["split_path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            self.data.append((tensor, r["class_id"]))

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor, label = self.data[idx]
        img = tensor.float() / 255.0
        
        if self.is_train:
            top = random.randint(0, 32)
            left = random.randint(0, 32)
            img = img[:, top:top+224, left:left+224]
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
        else:
            img = img[:, 16:16+224, 16:16+224]
            
        img = (img - self.mean) / self.std
        return img, label

train_ds = PreloadedDataset(train_records, is_train=True)
val_ds = PreloadedDataset(val_records, is_train=False)
test_ds = PreloadedDataset(test_records, is_train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Multi-Stage FitNets Distilled Neuravex
class FitNetNeuravex(nn.Module):
    def __init__(self, variant="edge", num_classes=6):
        super().__init__()
        self.neuravex = build_neuravex(variant, num_classes=num_classes)
        base_c = self.neuravex.base_c
        neck_c = base_c * 4
        in_dim = neck_c * 3  # q3, q4, q5
        
        # Intermediate feature projectors for training distillation (discarded at test time)
        self.proj3 = nn.Conv2d(base_c * 4, 512, 1)
        self.proj4 = nn.Conv2d(base_c * 8, 512, 1)
        self.proj5 = nn.Conv2d(base_c * 16, 512, 1)
        
        # Classification head with multi-scale attention gate
        self.channel_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_dim // 4, in_dim, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(0.2),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.early_exit_head = nn.Sequential(
            nn.BatchNorm1d(base_c * 8),
            nn.Linear(base_c * 8, num_classes)
        )

    def forward(self, x, return_projections=False):
        p3, p4, p5 = self.neuravex.backbone(x)
        f_early = p4.mean(dim=[-2, -1])
        logits_early = self.early_exit_head(f_early)
        
        q3, q4, q5 = self.neuravex.neck(p3, p4, p5)
        f3 = q3.mean(dim=[-2, -1])
        f4 = q4.mean(dim=[-2, -1])
        f5 = q5.mean(dim=[-2, -1])
        f_multi = torch.cat([f3, f4, f5], dim=1)
        f_calibrated = f_multi * self.channel_gate(f_multi)
        logits = self.classifier(f_calibrated)
        
        if return_projections:
            return logits, logits_early, self.proj3(p3), self.proj4(p4), self.proj5(p5)
        return logits, logits_early

    def switch_to_deploy(self):
        self.neuravex.switch_to_deploy()

model = FitNetNeuravex("edge", num_classes=6).to(device)

# Transfer stem weights from teacher
with torch.no_grad():
    yolo_stem_conv = teacher.model[0].conv
    yolo_stem_bn = teacher.model[0].bn
    c_out = model.neuravex.backbone.stem[0].conv.weight.shape[0]
    model.neuravex.backbone.stem[0].conv.weight.copy_(yolo_stem_conv.weight[:c_out, :3])
    model.neuravex.backbone.stem[0].bn.weight.copy_(yolo_stem_bn.weight[:c_out])
    model.neuravex.backbone.stem[0].bn.bias.copy_(yolo_stem_bn.bias[:c_out])
    model.neuravex.backbone.stem[0].bn.running_mean.copy_(yolo_stem_bn.running_mean[:c_out])
    model.neuravex.backbone.stem[0].bn.running_var.copy_(yolo_stem_bn.running_var[:c_out])

epochs = 120
optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

print("Starting Multi-Stage FitNets Distillation Training...")
best_val_acc = 0.0
best_test_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    total_loss, train_correct, train_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            t_out = teacher(imgs)
            t_logits = t_out[0] if isinstance(t_out, tuple) else t_out
            t_p3 = teacher_feats["p3"]
            t_p4 = teacher_feats["p4"]
            t_p5 = teacher_feats["p5"]
            
        optimizer.zero_grad()
        s_logits, s_early, sp3, sp4, sp5 = model(imgs, return_projections=True)
        
        # 1. Hard CE Loss
        loss_ce = F.cross_entropy(s_logits, labels, label_smoothing=0.05)
        loss_early = F.cross_entropy(s_early, labels, label_smoothing=0.05)
        
        # 2. Soft distillation loss (T=2.0)
        T = 2.0
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T * T)
        
        # 3. Multi-Stage Feature Mimicry Loss (Cosine + Normalized MSE across P3, P4, P5)
        def feature_loss(s_feat, t_feat):
            # Global pooled cosine similarity + normalized spatial MSE
            s_pool = s_feat.mean(dim=[-2, -1])
            t_pool = t_feat.mean(dim=[-2, -1])
            cos = 1.0 - F.cosine_similarity(s_pool, t_pool, dim=-1).mean()
            mse = F.mse_loss(F.normalize(s_feat, p=2, dim=1), F.normalize(t_feat, p=2, dim=1))
            return cos + mse

        loss_mimic = (feature_loss(sp3, t_p3) + feature_loss(sp4, t_p4) + feature_loss(sp5, t_p5)) / 3.0
        
        loss = loss_ce + 0.3 * loss_early + 0.7 * loss_kd + 0.2 * loss_mimic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        train_correct += (s_logits.argmax(dim=-1) == labels).sum().item()
        train_total += imgs.size(0)
        
    scheduler.step()
    
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, _ = model(imgs)
            val_correct += (logits.argmax(dim=-1) == labels).sum().item()
            val_total += imgs.size(0)
    val_acc = val_correct / val_total
    train_acc = train_correct / train_total
    
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits, _ = model(imgs)
                test_correct += (logits.argmax(dim=-1) == labels).sum().item()
                test_total += imgs.size(0)
        test_acc = test_correct / test_total
        best_test_acc = test_acc
        print(f"Epoch {epoch:3d}/{epochs}: Train: {train_acc*100:5.1f}%, Val: {val_acc*100:5.1f}% (BEST!), Test: {test_acc*100:5.1f}%")
    elif epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{epochs}: Train: {train_acc*100:5.1f}%, Val: {val_acc*100:5.1f}%")

print(f"\nFinal Best Val Acc: {best_val_acc*100:.2f}%, Test Acc: {best_test_acc*100:.2f}%")
