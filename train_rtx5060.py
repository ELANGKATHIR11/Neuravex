import time
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import build_yolo27
from neuravex.engine.trainer import neuravexMultiTaskTrainer
from neuravex.data.real_vegetable_dataset import RealVegetableDataset, real_dataset_collate_fn

def train_on_dgpu_rtx5060(epochs=5, batch_size=8, img_size=320):
    print("=" * 75)
    print("   TRAINING YOLO27 v0.6 ON NVIDIA GEFORCE RTX 5060 LAPTOP GPU")
    print("=" * 75)

    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Device: {device} ({gpu_name})")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Initial GPU Allocated Memory: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print(f"Initial GPU Reserved Memory:  {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB\n")

    # 1. Load Real Dataset from F:\Vegetable-Object-Detection
    train_dataset = RealVegetableDataset(root_dir="F:/Vegetable-Object-Detection", split="train", img_size=img_size, num_classes=4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=real_dataset_collate_fn
    )
    print(f"Loaded Real Dataset: {len(train_dataset)} training images across 4 classes.")
    print(f"Batches per epoch: {len(train_loader)} (Batch size = {batch_size})\n")

    # 2. Build YOLO27 v0.6 Model
    model = build_yolo27(size="small", num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = YOLO27MultiTaskTrainer(model, optimizer, device=device, num_classes=4, use_amp=True)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"AMP Mixed Precision: Enabled (GradScaler active)\n")

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):
            res = trainer.train_step(batch)
            loss_val = res["total_loss"]
            epoch_loss += loss_val
            epoch_batches += 1

            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx:2d}/{len(train_loader)}] | "
                      f"Total Loss: {loss_val:.4f} | GradNorm: {res['grad_norm']:.4f} | "
                      f"Det: {res['raw_losses'].get('det', 0):.4f} | "
                      f"Sem: {res['raw_losses'].get('semantic', 0):.4f} | "
                      f"Depth: {res['raw_losses'].get('depth', 0):.4f}")

        avg_loss = epoch_loss / max(epoch_batches, 1)
        epoch_time = time.perf_counter() - epoch_start
        print(f"--> Epoch {epoch} Completed in {epoch_time:.2f}s | Average Normalized Loss: {avg_loss:.4f}")
        print(f"    GPU Memory Peak: {torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB\n")

    total_time = time.perf_counter() - start_time
    print("=" * 75)
    print(f"TRAINING COMPLETE IN {total_time:.2f} SECONDS ON {gpu_name}!")
    print(f"Final Peak GPU Memory: {torch.cuda.max_memory_allocated(0) / (1024**2):.2f} MB")
    print("=" * 75)

    # Save real trained checkpoint
    os.makedirs("weights", exist_ok=True)
    save_path = "weights/yolo27_v06_rtx5060_vegetables.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "classes": ["Carrot", "Onion", "Potato", "Tomato"],
        "device": gpu_name
    }, save_path)
    print(f"Trained model checkpoint successfully saved to {save_path}!")

if __name__ == "__main__":
    train_on_dgpu_rtx5060(epochs=5, batch_size=8, img_size=320)
