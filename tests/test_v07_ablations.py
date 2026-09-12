import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import build_neuravex
from neuravex.engine.trainer import NeuravexMultiTaskTrainer
from neuravex.data.multitask_3d_dataset import Multitask3DDataset, multitask_3d_collate_fn

def run_v07_training_and_ablations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" NEURAVEX v0.7: FULL ABLATION SUITE & OVERFIT CONVERGENCE")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    dataset = Multitask3DDataset("data/synthetic_3d_dataset", split="train", img_size=320, num_classes=5)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=multitask_3d_collate_fn)

    # 1. Ablation Configurations
    ablations = [
        {"name": "Baseline (Supervised Only)", "ssl": False, "cross_geo": False, "tasks": None},
        {"name": "+ SSL (EMA Teacher + Distillation)", "ssl": True, "cross_geo": False, "tasks": None},
        {"name": "+ Cross-Task Geometry", "ssl": True, "cross_geo": True, "tasks": None},
        {"name": "Detection-First Fast Path", "ssl": True, "cross_geo": True, "tasks": ("det",)},
        {"name": "Full v0.7 Unified Multi-Task", "ssl": True, "cross_geo": True, "tasks": None}
    ]

    results = []

    for cfg in ablations:
        name = cfg["name"]
        print(f"\n--- Testing Ablation: {name} ---")
        model = build_neuravex(size="nano", num_classes=5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = NeuravexMultiTaskTrainer(
            model, optimizer, device=device, num_classes=5,
            use_amp=True, enable_ssl=cfg["ssl"]
        )

        initial_loss = None
        final_loss = None
        t0 = time.time()

        # Run 5 steps per ablation
        for step, batch in enumerate(loader):
            if step >= 5:
                break
            if cfg["tasks"] is not None:
                batch["task_masks"] = {"det": 1.0, "semantic": 0.0, "depth": 0.0, "geometry_3d": 0.0, "instance": 0.0, "boundary": 0.0, "mask_quality": 0.0, "consistency": 0.0, "ssl": 0.0, "cross_geo": 0.0}
            else:
                batch["task_masks"] = {"ssl": 1.0 if cfg["ssl"] else 0.0, "cross_geo": 1.0 if cfg["cross_geo"] else 0.0}

            step_res = trainer.train_step(batch)
            loss_val = step_res["total_loss"]
            if initial_loss is None:
                initial_loss = loss_val
            final_loss = loss_val

        dt = time.time() - t0
        throughput = 20 / dt # 20 samples processed
        results.append({
            "ablation": name,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "delta_loss": initial_loss - final_loss,
            "throughput_fps": throughput
        })
        print(f"    Initial: {initial_loss:.4f} -> Final: {final_loss:.4f} | Delta: {initial_loss - final_loss:.4f} | Throughput: {throughput:.1f} FPS")

    print("\n" + "=" * 80)
    print(" ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)
    for r in results:
        print(f" {r['ablation']:<35} | Init: {r['initial_loss']:6.4f} | Final: {r['final_loss']:6.4f} | Throughput: {r['throughput_fps']:5.1f} FPS")
    print("=" * 80)

if __name__ == "__main__":
    run_v07_training_and_ablations()
