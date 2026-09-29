"""
Training, Validation, and Testing CLI Script for the Animals Dataset.

Executes:
1. Stratified Train / Val / Test splitting on the dataset images
2. Unsupervised Object Discovery (Slot Attention) + Multi-view SSL Teacher training
3. Validation across epochs with early checkpoint saving
4. Final testing evaluation reporting SSL consistency loss, slot entropy, and objectness statistics
"""

import os
import sys
import argparse
import time
import json
import torch
from torch.utils.data import DataLoader

from neuravex.models.neuravex import build_neuravex
from neuravex.data.unsupervised_image_dataset import UnsupervisedImageFolderDataset
from neuravex.engine.unsupervised_trainer import NeuravexUnsupervisedTrainer


def run_pipeline(
    dataset_dir: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    img_size: int = 320,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints/animals_ssl",
):
    print("=" * 70)
    print(" NEURAVEX ANIMALS DATASET PIPELINE: TRAIN / VAL / TEST")
    print("=" * 70)
    print(f"Dataset:   {dataset_dir}")
    print(f"Device:    {device}")
    print(f"Epochs:    {epochs}")
    print(f"Batch Size:{batch_size}")
    print(f"Image Size:{img_size}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Datasets & DataLoaders
    print("\n[1/4] Preparing Train / Val / Test splits...")
    train_ds = UnsupervisedImageFolderDataset(
        dataset_dir, split="train", split_ratios=(0.7, 0.15, 0.15), img_size=img_size
    )
    val_ds = UnsupervisedImageFolderDataset(
        dataset_dir, split="val", split_ratios=(0.7, 0.15, 0.15), img_size=img_size
    )
    test_ds = UnsupervisedImageFolderDataset(
        dataset_dir, split="test", split_ratios=(0.7, 0.15, 0.15), img_size=img_size
    )

    print(f"  Classes found ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"  Train set: {len(train_ds)} images")
    print(f"  Val set:   {len(val_ds)} images")
    print(f"  Test set:  {len(test_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=("cuda" in device)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=("cuda" in device)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=("cuda" in device)
    )

    # 2. Build Neuravex Model with Slot Attention & Multi-Task capability
    print("\n[2/4] Constructing Neuravex architecture with Slot Attention...")
    model = build_neuravex(
        "nano",
        num_classes=len(train_ds.classes),
        enable_slots=True,
        num_slots=6,
        enable_pose=False,
        enable_st_intel=False,
    )

    trainer = NeuravexUnsupervisedTrainer(
        model=model,
        lr=lr,
        device=device,
        use_amp=("cuda" in device),
        ema_alpha=0.995,
    )

    # 3. Training Loop with Validation
    print("\n[3/4] Starting training loop...")
    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_animals_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        dt = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_ssl": round(train_metrics["ssl_loss"], 4),
            "train_slot": round(train_metrics["slot_loss"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_ssl": round(val_metrics["ssl_loss"], 4),
            "val_slot": round(val_metrics["slot_loss"], 4),
            "val_objectness": round(val_metrics["mean_slot_objectness"], 4),
            "duration_sec": round(dt, 2),
        }
        history.append(log_entry)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{dt:.1f}s] | "
            f"Train Loss: {log_entry['train_loss']:.4f} (SSL: {log_entry['train_ssl']:.4f}, Slot: {log_entry['train_slot']:.4f}) | "
            f"Val Loss: {log_entry['val_loss']:.4f} (SSL: {log_entry['val_ssl']:.4f}, Obj: {log_entry['val_objectness']:.3f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "classes": train_ds.classes,
                },
                best_checkpoint_path,
            )

    # 4. Testing
    print("\n[4/4] Evaluating on holdout Test set...")
    if os.path.exists(best_checkpoint_path):
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_loss: {ckpt['val_loss']:.4f})")

    test_metrics = trainer.evaluate(test_loader)
    print("\n" + "=" * 70)
    print(" FINAL TEST SET RESULTS")
    print("=" * 70)
    for k, v in test_metrics.items():
        print(f"  {k:25s}: {v:.4f}")

    results = {
        "dataset_dir": dataset_dir,
        "classes": train_ds.classes,
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "epochs": epochs,
        "history": history,
        "test_metrics": test_metrics,
        "best_checkpoint": best_checkpoint_path,
    }

    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull report and metrics written to: {report_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Val, Test Neuravex on Animals Dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"c:\Users\elang\Downloads\neuravex-cv\animals",
        help="Path to animals directory",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=320, help="Image resolution")
    parser.add_argument("--output-dir", type=str, default="checkpoints/animals_ssl", help="Output directory")

    args = parser.parse_args()
    run_pipeline(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        output_dir=args.output_dir,
    )
