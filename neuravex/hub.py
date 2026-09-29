"""
Neuravex Model Hub: Dynamic model zoo, download manager, and checkpoint loader.

Provides seamless programmatic and CLI downloading of pretrained Neuravex model
checkpoints and ONNX deployment graphs from GitHub releases and HuggingFace mirrors.
"""

import os
import sys
import urllib.request
import hashlib
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

# Base GitHub release URL for Neuravex asset downloads
GITHUB_RELEASE_TAG = "v0.2.0"
GITHUB_RELEASE_BASE = f"https://github.com/ELANGKATHIR11/Neuravex/releases/download/{GITHUB_RELEASE_TAG}"

# Model registry detailing variant, filename, size, tasks, and target device
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "neuravex-pico": {
        "filename": "neuravex_pico_weights.pth",
        "size_mb": 2.16,
        "variant": "pico",
        "description": "Ultra-lightweight micro model for microcontrollers & IoT edge (0.54M params)",
        "default_resolution": (320, 320),
        "target": "CPU / Raspberry Pi / MCU"
    },
    "neuravex-femto": {
        "filename": "neuravex_femto_weights.pth",
        "size_mb": 4.64,
        "variant": "femto",
        "description": "Femto edge model for low-power drones and mobile vision (1.16M params)",
        "default_resolution": (640, 640),
        "target": "Low-power Edge CPU / Jetson Nano"
    },
    "neuravex-nano": {
        "filename": "neuravex_nano_classifier_best.pth",
        "size_mb": 7.94,
        "variant": "nano",
        "description": "Pareto-optimal real-time nano perception model (2.02M params)",
        "default_resolution": (640, 640),
        "target": "Mobile / Jetson Orin / CPU & GPU"
    },
    "neuravex-nano-sota": {
        "filename": "sota_neuravex_nano_best.pth",
        "size_mb": 8.40,
        "variant": "nano",
        "description": "Distilled SOTA nano model with FitNets intermediate hint alignment",
        "default_resolution": (640, 640),
        "target": "Edge GPU / Workstation"
    },
    "neuravex-nano-onnx": {
        "filename": "sota_neuravex_nano.onnx",
        "size_mb": 4.12,
        "variant": "nano",
        "description": "Exported ONNX inference graph for Neuravex-Nano (FP32)",
        "default_resolution": (640, 640),
        "target": "ONNX Runtime (CPU/CUDA/TensorRT)"
    },
    "neuravex-nano-int8": {
        "filename": "sota_neuravex_nano_int8.onnx",
        "size_mb": 1.13,
        "variant": "nano",
        "description": "Quantized INT8 ONNX graph with dynamic range quantization",
        "default_resolution": (640, 640),
        "target": "ONNX Runtime INT8 Engine"
    },
    "neuravex-lite": {
        "filename": "neuravex_lite_weights.pth",
        "size_mb": 18.32,
        "variant": "lite",
        "description": "Balanced CPU/GPU model for smart cameras and robotics (4.58M params)",
        "default_resolution": (640, 640),
        "target": "Robotics / Smart Edge Devices"
    },
    "neuravex-edge": {
        "filename": "neuravex_edge_classifier_best.pth",
        "size_mb": 35.00,
        "variant": "edge",
        "description": "High-accuracy edge server model (9.08M params)",
        "default_resolution": (640, 640),
        "target": "Edge Servers / RTX 3050 / RTX 4060"
    },
    "neuravex-edge-sota": {
        "filename": "sota_neuravex_edge_best.pth",
        "size_mb": 36.05,
        "variant": "edge",
        "description": "Distilled SOTA edge model with dense multi-task fusion",
        "default_resolution": (640, 640),
        "target": "Workstation / Edge Server"
    },
    "neuravex-pro": {
        "filename": "neuravex_pro_weights.pth",
        "size_mb": 89.44,
        "variant": "pro",
        "description": "Full production workstation model for video analytics (22.36M params)",
        "default_resolution": (640, 640),
        "target": "Cloud / RTX 4090 / RTX 5060"
    },
    "neuravex-omni": {
        "filename": "neuravex_omni_foundation.pth",
        "size_mb": 143.84,
        "variant": "omni",
        "description": "Unified multimodal foundation stack (2D/3D/Depth/Seg/Pose/Track) (35.96M params)",
        "default_resolution": (640, 640),
        "target": "Cloud GPU / Multi-Camera Analytics"
    }
}


def get_cache_dir() -> str:
    """Return cache directory path for downloaded Neuravex models."""
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "neuravex", "models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def list_models():
    """Print the table of all downloadable Neuravex models."""
    print("=" * 95)
    print(f"{'Model Key':<24} | {'Variant':<8} | {'Size (MB)':<10} | {'Target Platform':<30} | {'Format'}")
    print("-" * 95)
    for key, info in MODEL_REGISTRY.items():
        fmt = "ONNX" if info["filename"].endswith(".onnx") else "PyTorch"
        print(f"{key:<24} | {info['variant']:<8} | {info['size_mb']:<10.2f} | {info['target']:<30} | {fmt}")
    print("=" * 95)


def download_model(model_name: str, dest_dir: Optional[str] = None, force: bool = False) -> str:
    """
    Download a pretrained model or checkpoint by its model key.

    Args:
        model_name: Model name key (e.g. 'neuravex-nano', 'neuravex-edge', 'neuravex-nano-int8')
        dest_dir: Optional custom destination folder. If None, uses ~/.cache/neuravex/models/
        force: If True, re-downloads even if file already exists locally.

    Returns:
        Local filepath to the downloaded model checkpoint.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        # Check if stripped of prefixes
        matched = None
        for k in MODEL_REGISTRY:
            if k == f"neuravex-{key}" or k.endswith(key):
                matched = k
                break
        if matched:
            key = matched
        else:
            valid_keys = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models:\n{valid_keys}")

    meta = MODEL_REGISTRY[key]
    filename = meta["filename"]
    save_folder = dest_dir or get_cache_dir()
    os.makedirs(save_folder, exist_ok=True)
    local_path = os.path.join(save_folder, filename)

    if os.path.exists(local_path) and not force and os.path.getsize(local_path) > 0:
        print(f"[Neuravex Hub] Using cached model: {local_path} ({os.path.getsize(local_path)/1e6:.2f} MB)")
        return local_path

    url = f"{GITHUB_RELEASE_BASE}/{filename}"
    print(f"[Neuravex Hub] Downloading {key} ({meta['size_mb']:.2f} MB) from:\n  -> {url}")
    print(f"  Destination: {local_path}")

    def progress_bar(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            bar = "#" * int(percent // 2) + "-" * (50 - int(percent // 2))
            sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({downloaded / 1e6:5.2f} / {total_size / 1e6:5.2f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, local_path, reporthook=progress_bar)
        print("\n[Neuravex Hub] Download complete successfully!")
    except Exception as e:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError(
            f"Failed downloading {filename} from {url}.\n"
            f"Error: {e}\n"
            f"Ensure your machine has internet access and the release asset is published at {GITHUB_RELEASE_BASE}"
        )

    return local_path


def load_model(model_name: str, num_classes: int = 80, device: Optional[str] = None, **kwargs) -> nn.Module:
    """
    Convenience method to download (if not cached) and instantiate a ready-to-infer Neuravex model.

    Args:
        model_name: Model identifier (e.g. 'neuravex-nano', 'neuravex-edge')
        num_classes: Number of detection classes (default: 80 for COCO)
        device: 'cuda', 'cpu', or None (auto-detects)
        **kwargs: Additional arguments forwarded to build_neuravex

    Returns:
        Instantiated and weighted torch.nn.Module in eval mode.
    """
    from .models.neuravex import build_neuravex

    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        for k in MODEL_REGISTRY:
            if k == f"neuravex-{key}" or k.endswith(key):
                key = k
                break

    meta = MODEL_REGISTRY.get(key, {})
    variant = meta.get("variant", "nano")
    weights_path = download_model(key)

    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model = build_neuravex(size=variant, num_classes=num_classes, **kwargs)
    ckpt = torch.load(weights_path, map_location=target_device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(target_device).eval()
    return model
