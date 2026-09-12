import time
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from yolo27.models.yolo27 import build_yolo27, YOLO27
from yolo27_v0_3 import YOLO27 as YOLO27_v04

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, x):
    """
    Standard FLOPs estimation for Conv2d, BatchNorm2d, Linear.
    """
    total_flops = 0
    def conv_hook(self, input, output):
        nonlocal total_flops
        b, c_out, h_out, w_out = output.shape
        c_in = self.in_channels
        k_h, k_w = self.kernel_size
        groups = self.groups
        flops = b * c_out * h_out * w_out * (2 * (c_in // groups) * k_h * k_w)
        total_flops += flops

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
    return total_flops

def benchmark_latency(model, x, num_warmup=10, num_runs=30):
    model.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_time_ms = ((end - start) / num_runs) * 1000.0
    fps = 1000.0 / avg_time_ms
    return avg_time_ms, fps

def run_benchmarks():
    print("=" * 70)
    print("      REAL BENCHMARK: YOLO27 v0.4 vs YOLO27 v0.5")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmark Device: {device}\n")

    # 1. Models setup
    model_v04 = YOLO27_v04().to(device)
    model_v05_nano = build_yolo27(size="nano").to(device)
    model_v05_small = build_yolo27(size="small").to(device)
    model_v05_medium = build_yolo27(size="medium").to(device)

    models = {
        "YOLO27 v0.4 (Baseline)": model_v04,
        "YOLO27 v0.5-Nano": model_v05_nano,
        "YOLO27 v0.5-Small": model_v05_small,
        "YOLO27 v0.5-Medium": model_v05_medium
    }

    # Parameter counts & sizes
    print(f"{'Model':<25} | {'Params (M)':<12} | {'Model Size (MB)':<15}")
    print("-" * 56)
    for name, m in models.items():
        total_p, _ = count_parameters(m)
        size_mb = (total_p * 4) / (1024 * 1024)
        print(f"{name:<25} | {total_p / 1e6:<12.2f} | {size_mb:<15.2f}")
    print()

    # Latency & FLOPs at 320x320 and 640x640
    resolutions = [320, 640]
    for res in resolutions:
        print(f"\n--- Resolution: {res}x{res} ---")
        print(f"{'Model':<25} | {'FLOPs (G)':<12} | {'Latency (ms)':<15} | {'FPS':<10}")
        print("-" * 68)

        for name, m in models.items():
            x = torch.rand(1, 3, res, res).to(device)
            flops = estimate_flops(m, x) / 1e9
            lat_ms, fps = benchmark_latency(m, x, num_warmup=5, num_runs=15)
            print(f"{name:<25} | {flops:<12.2f} | {lat_ms:<15.2f} | {fps:<10.1f}")

    print("\n" + "=" * 70)
    print("Benchmark complete. All numbers measured directly on local system.")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
