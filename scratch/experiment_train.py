import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path

# Paths
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

# Load manifest
manifest_path = ML_DIR / "publishable_benchmark" / "dataset_manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

classes = manifest["classes"]
train_records = [r for r in manifest["images"] if r["split"] == "train"]
val_records = [r for r in manifest["images"] if r["split"] == "val"]
test_records = [r for r in manifest["images"] if r["split"] == "test"]

print(f"Device: {device}")
print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

# Load teacher model
teacher_path = ML_DIR / "publishable_benchmark" / "weights" / "yolo_train_YOLO11m-cls" / "run" / "weights" / "best.pt"
teacher = YOLO(str(teacher_path)).model.to(device).eval()

# Hook teacher layer 4 (P3), layer 6 (P4), layer 9 (P5), and logits
teacher_feats = {}
def hook_fn(name):
    def hook(m, inp, outp):
        teacher_feats[name] = outp
    return hook

teacher.model[4].register_forward_hook(hook_fn("p3"))
teacher.model[6].register_forward_hook(hook_fn("p4"))
teacher.model[9].register_forward_hook(hook_fn("p5"))

class IndianAnimalsDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        import cv2
        self.images = []
        for r in records:
            img = cv2.imread(r["split_path"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append((img, r["class_id"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Strong, synchronized augmentations
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = IndianAnimalsDataset(train_records, transform=train_tf)
val_ds = IndianAnimalsDataset(val_records, transform=val_tf)
test_ds = IndianAnimalsDataset(test_records, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

# Let's test a student classifier
from publishable_benchmark.run_sota_optimization import SOTANeuravexClassifier

variant = "edge"
base_model = build_neuravex(variant, num_classes=6)

# Transfer stem weights from YOLO11m if channel dimensions permit
with torch.no_grad():
    yolo_stem_conv = teacher.model[0].conv
    yolo_stem_bn = teacher.model[0].bn
    c_out = base_model.backbone.stem[0].conv.weight.shape[0]
    base_model.backbone.stem[0].conv.weight.copy_(yolo_stem_conv.weight[:c_out, :3])
    base_model.backbone.stem[0].bn.weight.copy_(yolo_stem_bn.weight[:c_out])
    base_model.backbone.stem[0].bn.bias.copy_(yolo_stem_bn.bias[:c_out])
    base_model.backbone.stem[0].bn.running_mean.copy_(yolo_stem_bn.running_mean[:c_out])
    base_model.backbone.stem[0].bn.running_var.copy_(yolo_stem_bn.running_var[:c_out])
    print(f"Transferred {c_out} stem filters from YOLO11m!")

student = SOTANeuravexClassifier(base_model, num_classes=6, teacher_dim=512).to(device)

optimizer = optim.AdamW([
    {"params": student.neuravex.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
    {"params": student.channel_gate.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": student.teacher_proj.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": student.classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    {"params": student.early_exit_head.parameters(), "lr": 1e-3, "weight_decay": 1e-4}
])

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)

print("Starting training test...")
best_val_acc = 0.0

for epoch in range(1, 61):
    student.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Real-time online teacher forward pass
        with torch.no_grad():
            t_logits = teacher(imgs)
            if isinstance(t_logits, tuple):
                t_logits = t_logits[0]
            t_p5 = teacher_feats["p5"].mean(dim=[-2, -1])
            
        optimizer.zero_grad()
        s_logits, s_proj, s_early = student(imgs)
        
        # 1. Hard Cross-Entropy Loss
        loss_ce = F.cross_entropy(s_logits, labels, label_smoothing=0.05)
        loss_early = F.cross_entropy(s_early, labels, label_smoothing=0.05)
        
        # 2. Soft distillation loss (T=2.5)
        T = 2.5
        loss_kd = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean"
        ) * (T * T)
        
        # 3. Feature alignment loss (cosine sim with teacher penultimate pooled feature)
        cos_sim = F.cosine_similarity(s_proj, t_p5, dim=-1)
        loss_feat = (1.0 - cos_sim).mean()
        
        loss = loss_ce + 0.3 * loss_early + 0.7 * loss_kd + 0.1 * loss_feat
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        correct += (s_logits.argmax(dim=-1) == labels).sum().item()
        total += imgs.size(0)
        
    scheduler.step()
    
    # Evaluate on val
    student.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits, _, _ = student(imgs)
            val_correct += (logits.argmax(dim=-1) == labels).sum().item()
            val_total += imgs.size(0)
            
    val_acc = val_correct / val_total
    train_acc = correct / total
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Test on held-out test
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits, _, _ = student(imgs)
                test_correct += (logits.argmax(dim=-1) == labels).sum().item()
                test_total += imgs.size(0)
        test_acc = test_correct / test_total
        print(f"Epoch {epoch:2d}: Train Acc: {train_acc*100:5.1f}%, Val Acc: {val_acc*100:5.1f}% (BEST!), Test Acc: {test_acc*100:5.1f}%")
    elif epoch % 10 == 0:
        print(f"Epoch {epoch:2d}: Train Acc: {train_acc*100:5.1f}%, Val Acc: {val_acc*100:5.1f}%")

print(f"\nFinal Best Val Acc: {best_val_acc*100:.2f}%")
