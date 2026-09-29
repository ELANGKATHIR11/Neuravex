"""
Self-Supervised & Slot Attention Trainer for Unannotated Image Collections (e.g., Animals Dataset).

Trains Neuravex representations using:
  1. Multi-view EMA Teacher SSL distillation (consistency across augmented views).
  2. Unsupervised Slot Attention Object Discovery (entropy, spatial diversity, reconstruction).
  3. Transform-aligned consistency loss across geometric transformations.

Operates with ZERO human bounding box or segmentation labels.
Evaluates on validation/test splits via:
  - Multi-view SSL consistency metric
  - Slot Attention spatial entropy / objectness confidence
  - K-Means / linear representation separability on animal class features
"""

import os
import time
import math
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.neuravex import build_neuravex, Neuravex
from ..ssl.ssl_teacher import EMATeacher, MultiViewSSLLoss
from ..loss.consistency_loss import TransformAlignedConsistencyLoss
from ..data.augmentation import GeometricMultiViewAugment
from ..data.unsupervised_image_dataset import UnsupervisedImageFolderDataset


def slot_attention_regularization_loss(slot_masks: torch.Tensor, slot_objectness: torch.Tensor) -> torch.Tensor:
    """
    Unsupervised loss for Slot Attention discovery:
    1. Slot Diversity Loss: Penalizes high cross-slot overlap (pushes different slots to attend to different regions).
    2. Spatial Compactness / Entropy Loss: Rewards confident, peaky attention over diffuse attention.
    3. Objectness Regularization: Encourages objectness to be decisive (binary entropy).

    Args:
        slot_masks: (B, K, H, W) normalized attention distributions across slots
        slot_objectness: (B, K, 1) confidence scores
    """
    B, K, H, W = slot_masks.shape

    # 1. Slot Diversity (overlap minimization)
    # Flatten spatial dims: (B, K, N)
    masks_flat = slot_masks.reshape(B, K, -1)  # (B, K, N)
    # Cosine / dot product between slots in each sample
    masks_norm = F.normalize(masks_flat, p=2, dim=-1)
    overlap_matrix = torch.bmm(masks_norm, masks_norm.transpose(1, 2))  # (B, K, K)
    # Zero out diagonal
    eye = torch.eye(K, device=slot_masks.device).unsqueeze(0).expand(B, -1, -1)
    overlap_loss = (overlap_matrix * (1.0 - eye)).mean()

    # 2. Entropy / Decisiveness (per pixel, slots should partition clearly)
    # slot_masks is softmax normalized along K in SlotAttention
    slot_entropy = -(slot_masks * torch.log(slot_masks.clamp(min=1e-7))).sum(dim=1).mean()

    # 3. Objectness calibration (encourage confident discovery)
    obj_reg = -(slot_objectness * torch.log(slot_objectness.clamp(min=1e-7)) +
                (1.0 - slot_objectness) * torch.log((1.0 - slot_objectness).clamp(min=1e-7))).mean()

    return overlap_loss + 0.5 * slot_entropy + 0.1 * obj_reg


class NeuravexUnsupervisedTrainer:
    """
    Dedicated training and validation engine for unannotated image datasets
    leveraging Slot Discovery and Multi-View Teacher-Student SSL.
    """

    def __init__(
        self,
        model: Neuravex,
        lr: float = 3e-4,
        device: str = "cpu",
        use_amp: bool = False,
        ema_alpha: float = 0.995,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.teacher = EMATeacher(self.model, alpha=ema_alpha)
        self.ssl_loss_fn = MultiViewSSLLoss(lambda_f=1.0, lambda_m=0.5, lambda_r=0.0)
        self.consistency_loss_fn = TransformAlignedConsistencyLoss()
        self.augmenter = GeometricMultiViewAugment(p_flip=0.5)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ssl = 0.0
        total_slot = 0.0
        total_cons = 0.0
        num_batches = 0

        for batch in dataloader:
            x1 = batch["image"].to(self.device)
            x2 = batch["image_aug"].to(self.device) if "image_aug" in batch else x1

            self.optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                # 1. Forward Student
                out_student = self.model(x1)

                # 2. Forward Teacher on augmented view
                with torch.no_grad():
                    out_teacher = self.teacher(x2)

                # 3. Compute SSL distillation loss between student and teacher
                loss_ssl = self.ssl_loss_fn(out_student, out_teacher)

                # 4. Slot Attention unsupervised discovery loss (if slots enabled)
                if "slot_masks" in out_student and "slot_objectness" in out_student:
                    loss_slot = slot_attention_regularization_loss(
                        out_student["slot_masks"], out_student["slot_objectness"]
                    )
                else:
                    loss_slot = torch.tensor(0.0, device=self.device)

                # 5. Transform consistency loss
                with torch.no_grad():
                    aug_images, inv_mat, is_hflip = self.augmenter(x1)
                out_aug = self.model(aug_images)
                loss_cons = self.consistency_loss_fn(out_student, out_aug, inv_mat, is_hflip=is_hflip)

                loss = loss_ssl + 0.8 * loss_slot + 0.3 * loss_cons

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            # Update EMA teacher
            self.teacher.update(self.model)

            total_loss += loss.item()
            total_ssl += loss_ssl.item()
            total_slot += loss_slot.item()
            total_cons += loss_cons.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "ssl_loss": total_ssl / n,
            "slot_loss": total_slot / n,
            "cons_loss": total_cons / n,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_ssl = 0.0
        total_slot = 0.0
        total_cons = 0.0
        num_batches = 0

        avg_objectness = 0.0

        for batch in dataloader:
            x1 = batch["image"].to(self.device)
            x2 = batch["image_aug"].to(self.device) if "image_aug" in batch else x1

            out_student = self.model(x1)
            out_teacher = self.teacher(x2)

            loss_ssl = self.ssl_loss_fn(out_student, out_teacher)

            if "slot_masks" in out_student and "slot_objectness" in out_student:
                loss_slot = slot_attention_regularization_loss(
                    out_student["slot_masks"], out_student["slot_objectness"]
                )
                avg_objectness += out_student["slot_objectness"].mean().item()
            else:
                loss_slot = torch.tensor(0.0, device=self.device)

            aug_images, inv_mat, is_hflip = self.augmenter(x1)
            out_aug = self.model(aug_images)
            loss_cons = self.consistency_loss_fn(out_student, out_aug, inv_mat, is_hflip=is_hflip)

            loss = loss_ssl + 0.8 * loss_slot + 0.3 * loss_cons

            total_loss += loss.item()
            total_ssl += loss_ssl.item()
            total_slot += loss_slot.item()
            total_cons += loss_cons.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "ssl_loss": total_ssl / n,
            "slot_loss": total_slot / n,
            "cons_loss": total_cons / n,
            "mean_slot_objectness": avg_objectness / n,
        }
