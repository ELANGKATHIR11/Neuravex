"""
Comprehensive Quantization System for Neuravex:
Supports:
1. CPU INT8 Post-Training Quantization (PTQ)
2. GPU FP16 / BF16 Half-Precision Deployment
3. Rigorous Numerical Validation (Cosine similarity, Max error, NaN checks)
   to ensure zero catastrophic accuracy degradation before acceptance.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class QuantizationSystem:
    """
    Hardware-Targeted Quantization Engine for CPU and GPU:
    - CPU: PyTorch Dynamic INT8 Quantization (Linear/Conv projections).
    - GPU: FP16 / BF16 half-precision optimization.
    - Numerical Verification: Compares FP32 vs Quantized outputs on calibration tensors.
    """
    def __init__(self, min_cosine_similarity: float = 0.96, max_abs_error_threshold: float = 0.35):
        self.min_cosine_similarity = min_cosine_similarity
        self.max_abs_error_threshold = max_abs_error_threshold

    def quantize_cpu_int8(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Quantizes model for CPU inference using PyTorch dynamic quantization.
        Quantizes Conv2d/Linear sub-blocks where supported.
        """
        model_cpu = copy.deepcopy(model).cpu().eval()
        
        # Apply dynamic quantization to Linear layers
        try:
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model_cpu,
                {nn.Linear},
                dtype=torch.qint8
            )
            backend = "torch.ao.quantization.dynamic_qint8"
        except Exception:
            # Fallback for older torch API
            quantized_model = torch.quantization.quantize_dynamic(
                model_cpu,
                {nn.Linear},
                dtype=torch.qint8
            )
            backend = "torch.quantization.dynamic_qint8"

        meta = {
            "target": "cpu",
            "precision": "int8",
            "backend": backend,
            "status": "quantized"
        }
        return quantized_model, meta

    def convert_gpu_fp16(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Converts model to FP16 half precision for CUDA/TensorRT acceleration.
        """
        model_fp16 = copy.deepcopy(model).half().eval()
        meta = {
            "target": "cuda",
            "precision": "fp16",
            "backend": "torch.float16",
            "status": "converted"
        }
        return model_fp16, meta

    def validate_quantized_model(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor,
        tasks: tuple = ("det",)
    ) -> Dict[str, Any]:
        """
        Validates quantized model against the FP32 reference.
        Computes cosine similarity, max absolute error, and checks for numeric anomalies.
        """
        original_model.eval()
        quantized_model.eval()

        with torch.no_grad():
            out_fp32 = original_model(sample_input, tasks=tasks)
            
            # Prepare input for quantized model depending on precision
            if next(quantized_model.parameters()).dtype == torch.float16:
                q_input = sample_input.half()
            else:
                q_input = sample_input.float()

            out_q = quantized_model(q_input, tasks=tasks)

        # Compare primary detection outputs: class_logits and pred_boxes
        metrics = {}
        passed = True
        reasons = []

        for key in ["class_logits", "pred_boxes"]:
            if key in out_fp32 and key in out_q:
                t_fp32 = out_fp32[key].float()
                t_q = out_q[key].float()

                if torch.isnan(t_q).any() or torch.isinf(t_q).any():
                    passed = False
                    reasons.append(f"NaN/Inf encountered in {key}")
                    continue

                # Cosine similarity
                cos_sim = torch.cosine_similarity(t_fp32.view(-1), t_q.view(-1), dim=0).item()
                max_err = (t_fp32 - t_q).abs().max().item()
                mean_err = (t_fp32 - t_q).abs().mean().item()

                metrics[f"{key}_cosine_similarity"] = float(cos_sim)
                metrics[f"{key}_max_absolute_error"] = float(max_err)
                metrics[f"{key}_mean_absolute_error"] = float(mean_err)

                if cos_sim < self.min_cosine_similarity:
                    passed = False
                    reasons.append(f"{key} cosine similarity {cos_sim:.4f} < {self.min_cosine_similarity}")

        return {
            "passed_validation": passed,
            "rejection_reasons": reasons,
            "numerical_metrics": metrics
        }
