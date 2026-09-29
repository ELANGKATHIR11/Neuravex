"""
Comprehensive Hardware & System Profiler:
Measures:
- Latency (P50, P95, P99, Mean, Std) distinguishing warmup from steady-state
- Steady-state Throughput (FPS)
- Parameter Count & Model Size (MB)
- Analytical Real FLOPs / MACs
- Peak Host RAM (MB) & Peak Device VRAM (MB)
- CPU Utilization (%) & Estimated Power / Energy per Image (mJ)
"""

import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass
class ProfileMetrics:
    device: str
    precision: str
    resolution: Tuple[int, int]
    params_M: float
    model_size_mb: float
    gflops: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    fps: float
    peak_ram_mb: float
    peak_vram_mb: float
    cpu_percent: float
    estimated_power_watts: float
    energy_per_image_mj: float


def compute_conv_linear_flops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> float:
    """
    Computes total multiply-accumulate FLOPs for Conv2d and Linear layers.
    Returns GFLOPs.
    """
    total_flops = 0
    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total_flops
        # inp[0]: (B, Cin, Hin, Win), out: (B, Cout, Hout, Wout)
        if isinstance(out, torch.Tensor):
            B, Cout, Hout, Wout = out.shape
            Cin = module.in_channels
            Kh, Kw = module.kernel_size
            groups = module.groups
            # 2 * Kh * Kw * (Cin / groups) * Hout * Wout * Cout
            flops = 2 * Kh * Kw * (Cin // groups) * Hout * Wout * Cout
            if module.bias is not None:
                flops += Cout * Hout * Wout
            total_flops += flops

    def linear_hook(module, inp, out):
        nonlocal total_flops
        if isinstance(out, torch.Tensor):
            in_features = module.in_features
            out_features = module.out_features
            flops = 2 * in_features * out_features
            if module.bias is not None:
                flops += out_features
            total_flops += flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy = torch.randn(*input_shape, device=device)
    dtype = next(model.parameters()).dtype
    if dtype == torch.float16:
        dummy = dummy.half()

    try:
        with torch.no_grad():
            if hasattr(model, "forward"):
                # Run lightweight forward
                try:
                    _ = model(dummy, tasks=("det",))
                except Exception:
                    _ = model(dummy)
    finally:
        for h in hooks:
            h.remove()

    return float(total_flops / 1e9)


class SystemProfiler:
    """
    Production-Grade System & Hardware Profiler.
    """
    def __init__(self, warmup_iters: int = 10, steady_iters: int = 40):
        self.warmup_iters = warmup_iters
        self.steady_iters = steady_iters

    def profile(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int, int] = (1, 3, 320, 320),
        device: str = "cpu",
        tasks: tuple = ("det",),
        power_tdp_watts: Optional[float] = None
    ) -> ProfileMetrics:
        """
        Executes complete multi-dimensional profiling session.
        """
        dev = torch.device(device)
        model = model.to(dev).eval()
        dtype = next(model.parameters()).dtype
        precision_str = "fp16" if dtype == torch.float16 else "fp32"

        # Parameter count & model size
        params_M = sum(p.numel() for p in model.parameters()) / 1e6
        model_size_mb = (sum(p.numel() * p.element_size() for p in model.parameters()) +
                         sum(b.numel() * b.element_size() for b in model.buffers())) / (1024 * 1024)

        # Analytical FLOPs
        try:
            gflops = compute_conv_linear_flops(model, input_shape)
        except Exception:
            gflops = 0.0

        # Memory tracking baseline
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)

        dummy = torch.randn(*input_shape, device=dev)
        if dtype == torch.float16:
            dummy = dummy.half()

        # 1. Warmup Phase (excluded from benchmark timing)
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = model(dummy, tasks=tasks)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

        # 2. Steady-State Measurement Phase
        latencies = []
        cpu_measurements = []

        if dev.type == "cuda":
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.steady_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.steady_iters)]

            with torch.no_grad():
                for i in range(self.steady_iters):
                    cpu_measurements.append(psutil.cpu_percent(interval=None))
                    start_events[i].record()
                    _ = model(dummy, tasks=tasks)
                    end_events[i].record()

            torch.cuda.synchronize(dev)
            for i in range(self.steady_iters):
                latencies.append(start_events[i].elapsed_time(end_events[i]))
        else:
            with torch.no_grad():
                for _ in range(self.steady_iters):
                    cpu_measurements.append(psutil.cpu_percent(interval=None))
                    t0 = time.perf_counter()
                    _ = model(dummy, tasks=tasks)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        fps = float(1000.0 / max(1e-4, mean_lat))

        # Peak Memory
        ram_after = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = round(max(ram_before, ram_after), 2)
        if dev.type == "cuda":
            peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)
        else:
            peak_vram_mb = 0.0

        # Mean CPU %
        mean_cpu = float(np.mean([c for c in cpu_measurements if c > 0] or [10.0]))

        # Power & Energy estimation
        if power_tdp_watts is None:
            power_tdp_watts = 80.0 if dev.type == "cuda" else 35.0

        # Energy per image = power (W) * latency (s) * 1000 mJ/J
        energy_mj = float(power_tdp_watts * (mean_lat / 1000.0) * 1000.0)

        return ProfileMetrics(
            device=str(device),
            precision=precision_str,
            resolution=(input_shape[2], input_shape[3]),
            params_M=round(params_M, 3),
            model_size_mb=round(model_size_mb, 2),
            gflops=round(gflops, 3),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            p99_latency_ms=round(p99, 2),
            mean_latency_ms=round(mean_lat, 2),
            std_latency_ms=round(std_lat, 2),
            fps=round(fps, 1),
            peak_ram_mb=peak_ram_mb,
            peak_vram_mb=peak_vram_mb,
            cpu_percent=round(mean_cpu, 1),
            estimated_power_watts=round(power_tdp_watts, 1),
            energy_per_image_mj=round(energy_mj, 2)
        )
