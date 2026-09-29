"""
ONNX Export and ONNX Runtime Deployment Backend for Neuravex:
Provides:
1. Validated ONNX exporter supporting static and dynamic dimensions.
2. ORTInferenceEngine supporting CPUExecutionProvider, CUDAExecutionProvider,
   and TensorrtExecutionProvider with robust validation and automatic fallbacks.
3. Clean operator isolation and verification.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class NeuravexDetectionExportWrapper(nn.Module):
    """
    ONNX-friendly export wrapper that isolates the pure 2D detection graph.
    Returns (class_logits, pred_boxes) as standard tensors for maximum ONNX operator compatibility.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x, tasks=("det",), force_full_compute=True)
        return out["class_logits"], out["pred_boxes"]


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 320, 320),
    dynamic_axes: bool = True,
    opset_version: int = 14
) -> Dict[str, Any]:
    """
    Exports Neuravex detection model to an optimized ONNX file.
    Validates model integrity via onnx.checker.
    """
    import onnx

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    wrapper = NeuravexDetectionExportWrapper(model).eval()
    dummy_input = torch.randn(*input_shape)

    axes_config = None
    if dynamic_axes:
        axes_config = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "class_logits": {0: "batch_size", 1: "num_anchors"},
            "pred_boxes": {0: "batch_size", 1: "num_anchors"}
        }

    export_kwargs = {
        "model": wrapper,
        "args": dummy_input,
        "f": output_path,
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["class_logits", "pred_boxes"],
        "dynamic_axes": axes_config
    }
    # Pass dynamo=False if supported to avoid onnxscript requirement
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(**export_kwargs)

    # Verify ONNX model integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    return {
        "success": True,
        "output_path": output_path,
        "file_size_mb": float(file_size_mb),
        "opset_version": opset_version,
        "dynamic_axes": dynamic_axes
    }


class ORTInferenceEngine:
    """
    High-Performance ONNX Runtime Deployment Engine:
    Supports CPU, CUDA, and TensorRT execution providers with automated fallback.
    """
    def __init__(self, onnx_model_path: str, provider_preference: Optional[List[str]] = None):
        import onnxruntime as ort

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")

        self.onnx_model_path = onnx_model_path
        available = ort.get_available_providers()

        if provider_preference is None:
            # Default priority: CUDA -> CPU
            provider_preference = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Filter providers to those installed and available
        active_providers = [p for p in provider_preference if p in available]
        if "CPUExecutionProvider" not in active_providers:
            active_providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(onnx_model_path, sess_options, providers=active_providers)
            self.active_provider = self.session.get_providers()[0]
        except Exception as e:
            # Fallback strictly to CPU if CUDA/TRT failed to initialize
            self.session = ort.InferenceSession(onnx_model_path, sess_options, providers=["CPUExecutionProvider"])
            self.active_provider = "CPUExecutionProvider"

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, x: Any) -> Dict[str, np.ndarray]:
        """
        Runs inference via ONNX Runtime.
        x: torch.Tensor or numpy.ndarray
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        outputs = self.session.run(self.output_names, {self.input_name: x_np})
        return {
            "class_logits": outputs[0],
            "pred_boxes": outputs[1]
        }
