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
import torchvision.transforms as transforms
from pathlib import Path

REPO_ROOT = Path("c:/Users/elang/Downloads/neuravex-cv")
ML_DIR = REPO_ROOT / "ml_neuravex"
sys.path.insert(0, str(ML_DIR))

from neuravex.models.neuravex import build_neuravex
from ultralytics import YOLO
from publishable_benchmark.run_sota_optimization import SOTANeuravexClassifier

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
            # Resize to 256x256 to save memory and allow fast random crop
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            # Store as uint8 tensor [3, 256, 256]
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            self.data.append((tensor, r["class_id"]))

        # Augmentation transforms on torch tensors
        self.crop_size = 224
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor, label = self.data[idx]
        img = tensor.float() / 255.0
        
        if self.is_train:
            # Random crop 224 from 256
            top = random.randint(0, 32)
            left = random.randint(0, 32)
            img = img[:, top:top+224, left:left+224]
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
        else:
            # Center crop 224 from 256
            img = img[:, 16:16+224, 16:16+224]
            
        img = (img - self.mean) / self.std
        return img, label

train_ds = PreloadedDataset(train_records, is_train=True)
val_ds = PreloadedDataset(val_records, is_train=False)
test_ds = PreloadedDataset(test_records, is_train=False)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

print(f"Preloaded {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test images.")

# Test variant "edge" and "nano"
for variant in ["edge", "nano"]:
    print(f"\n{'='*60}\nTRAINING SOTA NEURAVEX-{variant.upper()}\n{'='*60}")
    base_model = build_neuravex(variant, num_classes=6)

    # Transfer stem weights from YOLO11m
    with torch.no_grad():
        yolo_stem_conv = teacher.model[0].conv
        yolo_stem_bn = teacher.model[0].bn
        c_out = base_model.backbone.stem[0].conv.weight.shape[0]
        base_model.backbone.stem[0].conv.weight.copy_(yolo_stem_conv.weight[:c_out, :3])
        base_model.backbone.stem[0].bn.weight.copy_(yolo_stem_bn.weight[:c_out])
        base_model.backbone.stem[0].bn.bias.copy_(yolo_stem_bn.bias[:c_out])
        base_model.backbone.stem[0].bn.running_mean.copy_(yolo_stem_bn.running_mean[:c_out])
        base_model.backbone.stem[0].bn.running_var.copy_(yolo_stem_bn.running_var[:c_out])

    student = SOTANeuravexClassifier(base_model, num_classes=6, teacher_dim=512).to(device)

    epochs = 100
    optimizer = optim.AdamW([
        {"params": student.neuravex.parameters(), "lr": 5e-4, "weight_decay": 1e-4},
        {"params": student.channel_gate.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": student.teacher_proj.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": student.classifier.parameters(), "lr": 1.5e-3, "weight_decay": 1e-4},
        {"params": student.early_exit_head.parameters(), "lr": 1.5e-3, "weight_decay": 1e-4}
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        student.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                t_logits = teacher(imgs)
                if isinstance(t_logits, tuple):
                    t_logits = t_logits[0]
                t_p5 = teacher_feats["p5"].mean(dim=[-2, -1])
                
            optimizer.zero_grad()
            s_logits, s_proj, s_early = student(imgs)
            
            # 1. Hard CE Loss
            loss_ce = F.cross_entropy(s_logits, labels, label_smoothing=0.05)
            loss_early = F.cross_entropy(s_early, labels, label_smoothing=0.05)
            
            # 2. Soft distillation (T=2.5)
            T = 2.5
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean"
            ) * (T * T)
            
            # 3. Feature cosine similarity
            cos_sim = F.cosine_similarity(s_proj, t_p5, dim=-1)
            loss_feat = (1.0 - cos_sim).mean()
            
            # 4. Relational distillation
            s_norm = F.normalize(s_proj, p=2, dim=-1)
            t_norm = F.normalize(t_p5, p=2, dim=-1)
            loss_rel = F.mse_loss(torch.mm(s_norm, s_norm.t()), torch.mm(t_norm, t_norm.t()))
            
            loss = loss_ce + 0.25 * loss_early + 0.6 * loss_kd + 0.05 * loss_feat + 0.05 * loss_rel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=4.0)
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (s_logits.argmax(dim=-1) == labels).sum().item()
            train_total += imgs.size(0)
            
        scheduler.step()
        
        # Validation
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
        train_acc = train_correct / train_total
        
        # Test evaluation
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    logits, _, _ = student(imgs)
                    test_correct += (logits.argmax(dim=-1) == labels).sum().item()
                    test_total += imgs.size(0)
            test_acc = test_correct / test_total
            best_test_acc = test_acc
            print(f"Epoch {epoch:3d}/{epochs}: Train: {train_acc*100:5.1f}%, Val: {val_acc*100:5.1f}% (BEST!), Test: {test_acc*100:5.1f}%")
        elif epoch % 20 == 0:
            print(f"Epoch {epoch:3d}/{epochs}: Train: {train_acc*100:5.1f}%, Val: {val_acc*100:5.1f}%")

    print(f"[{variant.upper()}] Final Peak Val Acc: {best_val_acc*100:.2f}%, Test Acc: {best_test_acc*100:.2f}%")
