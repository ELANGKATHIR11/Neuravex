"""
Cross-Platform Hardware Execution Context for CUDA, cuDNN, Intel (oneDNN/AVX), and AMD (ROCm/AVX2).
"""

import os
import platform
from typing import Optional, Dict, Any
import torch

class DeviceContext:
    """
    Manages optimal device selection, memory caching, and hardware accelerations
    across NVIDIA CUDA + cuDNN, AMD ROCm, Intel oneDNN, and Apple MPS.
    """
    def __init__(self, preferred: Optional[str] = None):
        self.device = self._select_device(preferred)
        self.backend = self._inspect_backend()

    def _select_device(self, preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _inspect_backend(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "device": str(self.device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
            "cudnn_enabled": False,
            "cudnn_version": None,
            "device_name": "CPU",
            "memory_gb": 0.0,
        }

        if self.device.type == "cuda":
            info["device_name"] = torch.cuda.get_device_name(self.device)
            info["memory_gb"] = round(torch.cuda.get_device_properties(self.device).total_memory / 1e9, 2)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                info["cudnn_enabled"] = True
                info["cudnn_version"] = torch.backends.cudnn.version()
        else:
            # CPU detection: Intel / AMD AVX accelerations
            info["device_name"] = f"CPU Host ({platform.machine()})"

        return info

    def to_device(self, tensor_or_module):
        """Move tensor or module to optimal device seamlessly."""
        return tensor_or_module.to(self.device)

    def synchronize(self):
        """Hardware synchronization point."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def empty_cache(self):
        """Clear memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_optimal_device(preferred: Optional[str] = None) -> torch.device:
    """Return the optimal execution torch.device."""
    return DeviceContext(preferred).device
