"""
Hardware-Aware Controller for Neuravex:
Profiles system hardware capabilities (CPU cores, RAM, GPU architecture, VRAM)
and dynamically configures:
1. Optimal model variant (Pico -> Femto -> Nano -> Lite -> Edge -> Pro -> Omni)
2. Precision (INT8, FP16, FP32)
3. Input resolution (320, 512, 640)
4. Dynamic routing budgets
5. Optimal execution backend (PyTorch CPU/CUDA, ONNX Runtime CPU/CUDA)
"""

import os
import psutil
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HardwareProfile:
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_available: bool
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_compute_capability: Optional[str]


@dataclass
class DeploymentRecommendation:
    model_variant: str
    precision: str
    input_resolution: int
    backend: str
    routing_budget: str
    rationale: str


class HardwareAwareController:
    """
    Intelligent System Profiler & Hardware-Adaptive Deployment Controller.
    """
    def __init__(self):
        self.profile = self.profile_system()

    def profile_system(self) -> HardwareProfile:
        """Discovers host hardware specs."""
        phys_cores = psutil.cpu_count(logical=False) or 4
        logical_cores = psutil.cpu_count(logical=True) or phys_cores
        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024 ** 3)
        ram_available_gb = ram.available / (1024 ** 3)

        gpu_available = torch.cuda.is_available()
        gpu_name = "None"
        vram_total_gb = 0.0
        vram_free_gb = 0.0
        cuda_cc = None

        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                free_bytes, _ = torch.cuda.mem_get_info(0)
                vram_free_gb = free_bytes / (1024 ** 3)
                major, minor = torch.cuda.get_device_capability(0)
                cuda_cc = f"{major}.{minor}"
            except Exception:
                gpu_available = False

        return HardwareProfile(
            cpu_cores_physical=phys_cores,
            cpu_cores_logical=logical_cores,
            ram_total_gb=round(ram_total_gb, 2),
            ram_available_gb=round(ram_available_gb, 2),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            vram_total_gb=round(vram_total_gb, 2),
            vram_free_gb=round(vram_free_gb, 2),
            cuda_compute_capability=cuda_cc
        )

    def recommend(self) -> DeploymentRecommendation:
        """
        Synthesizes hardware metrics into optimal deployment parameters.
        """
        p = self.profile
        # Case 1: High/Medium Modern GPU (e.g. RTX 5060, RTX 4060, RTX 3060 with >= 6GB VRAM)
        if p.gpu_available and p.vram_total_gb >= 6.0:
            return DeploymentRecommendation(
                model_variant="edge",
                precision="fp16",
                input_resolution=640,
                backend="onnxruntime_cuda",
                routing_budget="high_throughput_gpu",
                rationale=f"Detected high-capability GPU ({p.gpu_name}, {p.vram_total_gb:.1f}GB VRAM). Configured for FP16 CUDA acceleration."
            )
        
        # Case 2: Low-VRAM / Mobile Edge GPU (< 6GB VRAM, e.g. Jetson / GTX 1650 / RTX 3050 mobile)
        elif p.gpu_available and p.vram_total_gb < 6.0:
            return DeploymentRecommendation(
                model_variant="nano",
                precision="fp16",
                input_resolution=512,
                backend="onnxruntime_cuda",
                routing_budget="constrained_gpu",
                rationale=f"Detected entry/edge GPU ({p.gpu_name}, {p.vram_total_gb:.1f}GB VRAM). Configured for lightweight FP16 inference."
            )
            
        # Case 3: Desktop CPU (Multi-core, >= 12GB RAM)
        elif p.cpu_cores_physical >= 6 and p.ram_available_gb >= 8.0:
            return DeploymentRecommendation(
                model_variant="lite",
                precision="int8",
                input_resolution=512,
                backend="onnxruntime_cpu",
                routing_budget="cpu_high_efficiency",
                rationale=f"Detected workstation CPU ({p.cpu_cores_physical} physical cores, {p.ram_available_gb:.1f}GB free RAM). Configured for multi-threaded INT8 execution."
            )

        # Case 4: Constrained / IoT / Edge CPU
        else:
            return DeploymentRecommendation(
                model_variant="pico",
                precision="int8",
                input_resolution=320,
                backend="onnxruntime_cpu",
                routing_budget="cpu_ultra_lightweight",
                rationale=f"Detected resource-constrained host ({p.cpu_cores_physical} cores). Selected Pico variant with 320x320 resolution for real-time CPU FPS."
            )

    def get_hardware_report(self) -> Dict[str, Any]:
        """Returns JSON-serializable hardware assessment and deployment recommendations."""
        rec = self.recommend()
        return {
            "hardware_profile": {
                "cpu_cores_physical": self.profile.cpu_cores_physical,
                "cpu_cores_logical": self.profile.cpu_cores_logical,
                "ram_total_gb": self.profile.ram_total_gb,
                "ram_available_gb": self.profile.ram_available_gb,
                "gpu_available": self.profile.gpu_available,
                "gpu_name": self.profile.gpu_name,
                "vram_total_gb": self.profile.vram_total_gb,
                "vram_free_gb": self.profile.vram_free_gb,
                "cuda_compute_capability": self.profile.cuda_compute_capability
            },
            "recommendation": {
                "model_variant": rec.model_variant,
                "precision": rec.precision,
                "input_resolution": rec.input_resolution,
                "backend": rec.backend,
                "routing_budget": rec.routing_budget,
                "rationale": rec.rationale
            }
        }
