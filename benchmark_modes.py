"""
Neuravex Benchmark Suite:
Separately profiles:
  1. DET-only (tasks=('det',))
  2. DET+SEG (tasks=('det', 'instance'))
  3. Full 3D Multi-task (tasks=None)
Reports:
  - Latency (P50, P95, Mean in ms)
  - Throughput (FPS)
  - Peak VRAM (if GPU available) or RAM overhead
  - Parameter counts & FLOPs estimate
"""

import time
import torch
import numpy as np
from neuravex import build_neuravex, CameraIntrinsics

def profile_task_mode(model, x, K, tasks, warmup=15, runs=50, device="cpu"):
    latencies = []
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x, tasks=tasks, intrinsics=K)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for _ in range(runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, tasks=tasks, intrinsics=K)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)
    fps = 1000.0 / mean_lat

    vram_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "mean_ms": float(mean_lat),
        "fps": float(fps),
        "vram_mb": float(vram_mb)
    }

def count_parameters(model, tasks=None):
    if tasks == ("det",):
        # Active modules: backbone, neck, router_p3, det_head
        modules = [model.backbone, model.neck, model.router_p3, model.det_head]
        params = sum(sum(p.numel() for p in m.parameters()) for m in modules)
    elif tasks == ("det", "instance"):
        modules = [model.backbone, model.neck, model.router_p3, model.det_head, model.seg_head]
        params = sum(sum(p.numel() for p in m.parameters()) for m in modules)
    else:
        params = sum(p.numel() for p in model.parameters())
    return params

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing Neuravex Benchmark on device: {device.upper()}")

    model = build_neuravex(size="small", num_classes=80)
    model.eval()
    model.to(device)

    K = CameraIntrinsics(fx=721.5, fy=721.5, cx=320.0, cy=320.0)
    x = torch.randn(1, 3, 640, 640, device=device)

    modes = [
        ("DET-only [tasks=('det',)]", ("det",)),
        ("DET+SEG [tasks=('det', 'instance')]", ("det", "instance")),
        ("Full 3D Multitask [tasks=None]", None),
    ]

    print("\n" + "="*80)
    print(f"{'Mode':<36} | {'Params (M)':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'FPS':<8} | {'VRAM (MB)'}")
    print("="*80)

    for name, tasks in modes:
        params = count_parameters(model, tasks=tasks) / 1e6
        res = profile_task_mode(model, x, K, tasks, warmup=10, runs=30, device=device)
        print(f"{name:<36} | {params:<10.2f} | {res['p50_ms']:<9.2f} | {res['p95_ms']:<9.2f} | {res['fps']:<8.1f} | {res['vram_mb']:<8.1f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
