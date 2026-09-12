import time
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import build_yolo27
from neuravex.engine.trainer import neuravexMultiTaskTrainer
from neuravex.data.multitask_3d_dataset import Multitask3DDataset, multitask_3d_collate_fn

def train_3d_and_dem_rtx5060(epochs=6, batch_size=8, img_size=320):
    print("=" * 80)
    print(" TRAINING YOLO27 v0.6 FOR 3D BOUNDING BOXES (XYZ, LWH, YAW) & DEM ON RTX 5060")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")

    # 1. Dataset
    train_dataset = Multitask3DDataset(data_dir="data/synthetic_3d_dataset", split="train", img_size=img_size, num_classes=5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multitask_3d_collate_fn
    )
    print(f"Loaded 3D + DEM Dataset: {len(train_dataset)} training samples.")
    print(f"Batches per epoch: {len(train_loader)} (Batch size = {batch_size})\n")

    # 2. Build Model
    model = build_yolo27(size="small", num_classes=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = YOLO27MultiTaskTrainer(model, optimizer, device=device, num_classes=5, use_amp=True)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print("All Modalities Active: 2D Det, 3D Box (XYZ, LWH, Yaw), Oriented 3D IoU, DEM Metric Depth\n")

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            res = trainer.train_step(batch)
            loss_val = res["total_loss"]
            epoch_loss += loss_val

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx:2d}/{len(train_loader)}] | "
                      f"Loss: {loss_val:.4f} | "
                      f"3D Loss: {res['raw_losses'].get('geometry_3d', 0):.4f} | "
                      f"DEM Depth: {res['raw_losses'].get('depth', 0):.4f} | "
                      f"Det: {res['raw_losses'].get('det', 0):.4f}")

        avg_loss = epoch_loss / len(train_loader)
        epoch_dur = time.perf_counter() - epoch_start
        print(f"--> Epoch {epoch} Completed in {epoch_dur:.2f}s | Average Normalized Loss: {avg_loss:.4f}")
        if torch.cuda.is_available():
            print(f"    Peak VRAM: {torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB\n")

    total_dur = time.perf_counter() - start_time
    print("=" * 80)
    print(f"3D + DEM TRAINING COMPLETE IN {total_dur:.2f}s ON {gpu_name}!")
    print("=" * 80)

    # Save Checkpoint
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/yolo27_v06_rtx5060_3d_dem.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "modalities": ["2d_boxes", "3d_xyz", "3d_lwh", "3d_yaw", "dem_depth", "semantic", "boundary"],
        "device": gpu_name
    }, save_path)
    print(f"Saved trained 3D+DEM model checkpoint to {save_path}!")

if __name__ == "__main__":
    train_3d_and_dem_rtx5060(epochs=6, batch_size=8, img_size=320)
