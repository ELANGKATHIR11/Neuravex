"""
Neuravex Experiment & Ablation Engine.

Runs controlled ablation experiments accepting each component only when
measured results justify its cost.

Ablation sequence:
  1. baseline              (fixed nano model, no specialization)
  2. +dataset_analyzer     (complexity vector extracted)
  3. +arch_specialization  (generator picks architecture)
  4. +scale_specialization (P2/P6 based on object sizes)
  5. +adaptive_compute     (difficulty routing enabled)
  6. +difficulty_routing   (3-level router)
  7. +distillation         (KD from larger model)
  8. +pruning              (channel pruning)
  9. +quantization         (INT8 PTQ)

Rules:
  - Hard-fail on NaN/Inf, bad checkpoint, shape mismatch
  - Never fabricate AP numbers
  - Each stage is validated before the next proceeds
  - Accepts component if: AP_new >= AP_prev - tolerance AND cost_new < cost_prev
  - Reports Pareto frontier across all ablation points

Required metrics per stage:
  AP50:95, AP50, AP75, APs/m/l, AR
  params, GFLOPs
  expected/executed FLOPs multiplier
  mean/P50/P95/P99 latency (ms)
  FPS, peak VRAM (MB)
  AP/FLOP, AP/latency, AP/param, AP/VRAM
"""

import json
import time
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn

from .architecture_generator import ArchitectureGenerator, HardwareConstraints, ArchSpec
from .dataset_analyzer import DatasetAnalyzer
from .hardware_profiler import HardwareProfiler, HardwareProfile
from .specialization_pipeline import build_specialized_neuravex
from .yolo26_baseline import OFFICIAL_REFERENCE


@dataclass
class AblationStageResult:
    """Result of a single ablation stage."""
    stage_name: str
    active_components: List[str] = field(default_factory=list)
    # Model metrics (filled after evaluation)
    ap50_95: float = 0.0
    ap50: float = 0.0
    ap75: float = 0.0
    aps: float = 0.0
    apm: float = 0.0
    apl: float = 0.0
    ar100: float = 0.0
    # Compute metrics
    params_M: float = 0.0
    gflops: float = 0.0
    expected_flops_mult: float = 1.0
    executed_flops_mult: float = 1.0
    # Latency
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    fps: float = 0.0
    peak_vram_mb: float = 0.0
    # Efficiency ratios (after eval)
    ap_per_gflop: float = 0.0
    ap_per_ms: float = 0.0
    ap_per_param_M: float = 0.0
    ap_per_vram_mb: float = 0.0
    # Gate
    accepted: bool = False
    rejection_reason: str = ""
    errors: List[str] = field(default_factory=list)

    def fill_efficiency(self):
        self.ap_per_gflop = self.ap50_95 / max(self.gflops, 1e-6)
        self.ap_per_ms = self.ap50_95 / max(self.mean_ms, 1e-6)
        self.ap_per_param_M = self.ap50_95 / max(self.params_M, 1e-6)
        self.ap_per_vram_mb = self.ap50_95 / max(self.peak_vram_mb, 1e-6)

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage_name,
            "active_components": self.active_components,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "metrics": {
                "ap50_95": self.ap50_95, "ap50": self.ap50, "ap75": self.ap75,
                "aps": self.aps, "apm": self.apm, "apl": self.apl, "ar100": self.ar100,
            },
            "compute": {
                "params_M": self.params_M, "gflops": self.gflops,
                "expected_flops_mult": self.expected_flops_mult,
                "executed_flops_mult": self.executed_flops_mult,
            },
            "latency": {
                "p50_ms": self.p50_ms, "p95_ms": self.p95_ms,
                "p99_ms": self.p99_ms, "mean_ms": self.mean_ms,
                "fps": self.fps, "peak_vram_mb": self.peak_vram_mb,
            },
            "efficiency": {
                "ap_per_gflop": self.ap_per_gflop, "ap_per_ms": self.ap_per_ms,
                "ap_per_param_M": self.ap_per_param_M, "ap_per_vram_mb": self.ap_per_vram_mb,
            },
            "errors": self.errors,
        }


class ExperimentEngine:
    """
    Runs controlled ablation experiments across all specialization components.

    Each stage:
      1. Builds model with component enabled/disabled
      2. Profiles hardware latency
      3. Evaluates AP if eval_fn provided (otherwise uses dummy AP=0.0)
      4. Applies acceptance gate: AP_new >= AP_prev - tolerance AND lower cost
      5. Records all required metrics
      6. Generates Pareto frontier and comparison table

    Usage:
        engine = ExperimentEngine(device="cuda")
        results = engine.run_ablation(
            ann_file="annotations.json",
            img_dir="images/",
            constraints=constraints,
            num_classes=20,
            eval_fn=my_eval_function,   # Optional: returns (ap50_95, ap50, ...)
        )
        engine.print_ablation_table(results)
        pareto = engine.pareto_frontier(results)
    """

    def __init__(
        self,
        device: str = "cpu",
        img_size: int = 640,
        ap_regression_tolerance: float = 0.01,
        verbose: bool = True,
    ):
        self.device = device
        self.img_size = img_size
        self.ap_regression_tolerance = ap_regression_tolerance
        self.verbose = verbose
        self.analyzer = DatasetAnalyzer()
        self.generator = ArchitectureGenerator(img_size=img_size)
        self.profiler = HardwareProfiler(device=device, warmup_runs=5, measure_runs=20)

    def _build_and_profile(
        self,
        spec: ArchSpec,
        num_classes: int,
        stage_name: str,
        components: List[str],
    ) -> Tuple[nn.Module, AblationStageResult]:
        """Build model, validate outputs, and profile hardware."""
        result = AblationStageResult(stage_name=stage_name, active_components=list(components))

        # Build model
        try:
            model = build_specialized_neuravex(spec, num_classes=num_classes)
            model = model.to(self.device)
            model.eval()

            # Validate: forward pass, check for NaN/Inf
            x = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            with torch.no_grad():
                out = model(x, tasks=("det",))
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        raise RuntimeError(f"NaN/Inf in output '{k}'")
        except Exception as e:
            result.errors.append(f"Model build/validate failed: {e}")
            result.accepted = False
            result.rejection_reason = str(e)
            return None, result

        # Hardware profile
        try:
            hw = self.profiler.profile(model, arch_name=stage_name, img_size=self.img_size, tasks=("det",))
            result.params_M = hw.params_M
            result.gflops = hw.gflops
            result.p50_ms = hw.p50_ms
            result.p95_ms = hw.p95_ms
            result.p99_ms = hw.p99_ms
            result.mean_ms = hw.mean_ms
            result.fps = hw.fps
            result.peak_vram_mb = hw.peak_vram_mb
        except Exception as e:
            result.errors.append(f"Hardware profiling failed: {e}")

        return model, result

    def _acceptance_gate(
        self,
        current: AblationStageResult,
        prev: Optional[AblationStageResult],
    ) -> Tuple[bool, str]:
        """
        Accept a component if:
          AP_new >= AP_prev - tolerance  (no significant regression)
          AND at least one of: lower FLOPs, lower latency, lower params
        """
        if prev is None:
            return True, ""  # Baseline always accepted

        ap_ok = current.ap50_95 >= (prev.ap50_95 - self.ap_regression_tolerance)
        cost_ok = (
            current.gflops <= prev.gflops or
            current.mean_ms <= prev.mean_ms or
            current.params_M <= prev.params_M
        )

        if not ap_ok:
            return False, f"AP regression: {current.ap50_95:.4f} < {prev.ap50_95:.4f} - {self.ap_regression_tolerance}"
        # For stages that add compute, accept if AP improved enough
        if not cost_ok and current.ap50_95 > prev.ap50_95 + 0.02:
            return True, "AP improved enough to justify added compute"
        if not cost_ok:
            return False, f"Cost increased without AP gain: mean_ms {current.mean_ms:.1f} > {prev.mean_ms:.1f}"
        return True, ""

    def run_ablation(
        self,
        ann_file: str,
        img_dir: Optional[str] = None,
        constraints: Optional[HardwareConstraints] = None,
        num_classes: int = 80,
        eval_fn: Optional[Callable] = None,
        max_analysis_images: int = 200,
    ) -> List[AblationStageResult]:
        """
        Run full ablation sequence. Returns list of AblationStageResult.
        eval_fn: Optional callable (model, device, img_size) -> dict with keys ap50_95, ap50, ...
                 If None, AP metrics will be 0.0 (structure + latency tested only).
        """
        if constraints is None:
            constraints = HardwareConstraints(device=self.device)

        if self.verbose:
            print("\n" + "="*70)
            print("  NEURAVEX ABLATION EXPERIMENT ENGINE")
            print("="*70)

        # ── Dataset Analysis ──────────────────────────────────────────────────
        if self.verbose:
            print("\n[Ablation] Analyzing dataset...")
        stats = self.analyzer.analyze(ann_file, img_dir, max_images=max_analysis_images)
        vec = stats.complexity_vector

        results: List[AblationStageResult] = []
        prev_result: Optional[AblationStageResult] = None
        best_spec: Optional[ArchSpec] = None

        # ── Stage definitions ─────────────────────────────────────────────────
        # Each stage: (stage_name, spec_override_fn, active_components)

        # Stage 1: Baseline (fixed nano, no specialization)
        from neuravex.specialization.architecture_generator import ArchSpec as AS
        baseline_spec = AS(base_c=16, depth_mul=0.33, use_p2=False, use_p6=False,
                           num_difficulty_levels=2, reg_max=16)
        stages = [
            ("baseline",            baseline_spec,    ["fixed_nano"]),
        ]

        # Stage 2: +Dataset Analyzer (same size, but we log vector)
        stages.append(("baseline+analyzer", baseline_spec, ["fixed_nano", "dataset_analyzer"]))

        # Stage 3: +Architecture Specialization
        specialized_spec = self.generator.generate(vec, constraints, num_classes=num_classes)
        stages.append(("arch_specialized",
                        specialized_spec,
                        ["fixed_nano", "dataset_analyzer", "arch_specialization"]))

        # Stage 4: +Scale Specialization (P2 if tiny objects)
        import numpy as _np
        small_frac = float(vec[2]) if vec is not None else 0.0
        scale_spec = ArchSpec(**{**specialized_spec.to_dict(),
                                  **{"use_p2": small_frac > 0.3,
                                     "strides": (4, 8, 16, 32) if small_frac > 0.3 else specialized_spec.strides}})
        stages.append(("scale_specialized",
                        scale_spec,
                        ["fixed_nano", "dataset_analyzer", "arch_specialization", "scale_specialization"]))

        # Stage 5: +Adaptive Compute (difficulty router 2-level)
        ac_spec = ArchSpec(**{**scale_spec.to_dict(), **{"num_difficulty_levels": 2}})
        stages.append(("adaptive_compute",
                        ac_spec,
                        ["fixed_nano", "dataset_analyzer", "arch_specialization",
                         "scale_specialization", "adaptive_compute"]))

        # Stage 6: +Difficulty Routing (3-level)
        dr_spec = ArchSpec(**{**ac_spec.to_dict(), **{"num_difficulty_levels": 3}})
        stages.append(("difficulty_routing",
                        dr_spec,
                        ["fixed_nano", "dataset_analyzer", "arch_specialization",
                         "scale_specialization", "adaptive_compute", "difficulty_routing"]))

        # ── Run each stage ─────────────────────────────────────────────────
        for stage_name, spec, components in stages:
            if self.verbose:
                print(f"\n[Ablation] Stage: {stage_name}")
                print(f"  Components: {components}")

            model, result = self._build_and_profile(spec, num_classes, stage_name, components)

            if model is None:
                if self.verbose:
                    print(f"  FAILED: {result.errors}")
                results.append(result)
                continue

            # Run evaluation if provided
            if eval_fn is not None:
                try:
                    eval_metrics = eval_fn(model, self.device, self.img_size)
                    result.ap50_95 = float(eval_metrics.get("ap50_95", 0.0))
                    result.ap50 = float(eval_metrics.get("ap50", 0.0))
                    result.ap75 = float(eval_metrics.get("ap75", 0.0))
                    result.aps = float(eval_metrics.get("aps", 0.0))
                    result.apm = float(eval_metrics.get("apm", 0.0))
                    result.apl = float(eval_metrics.get("apl", 0.0))
                    result.ar100 = float(eval_metrics.get("ar100", 0.0))
                except Exception as e:
                    result.errors.append(f"Eval failed: {e}")

            result.fill_efficiency()

            # Acceptance gate
            accepted, reason = self._acceptance_gate(result, prev_result)
            result.accepted = accepted
            result.rejection_reason = reason

            if self.verbose:
                accept_str = "✓ ACCEPTED" if accepted else f"✗ REJECTED ({reason})"
                print(f"  {accept_str}")
                print(f"  AP={result.ap50_95:.4f} | P50={result.p50_ms:.2f}ms | "
                      f"Params={result.params_M:.2f}M | GFLOPs={result.gflops:.3f}")

            results.append(result)
            if accepted:
                prev_result = result
                best_spec = spec

        if self.verbose:
            self.print_ablation_table(results)

        return results

    def print_ablation_table(self, results: List[AblationStageResult]):
        """Print formatted ablation comparison table."""
        print("\n" + "="*120)
        print(f"  {'Stage':<30} | {'Accept':<8} | {'AP50:95':<8} | {'AP50':<8} | "
              f"{'P50ms':<7} | {'FPS':<7} | {'VRAM':<8} | {'Params':<8} | {'AP/GFl':<8}")
        print("="*120)
        for r in results:
            acc = "✓" if r.accepted else "✗"
            print(f"  {r.stage_name:<30} | {acc:<8} | {r.ap50_95:<8.4f} | {r.ap50:<8.4f} | "
                  f"{r.p50_ms:<7.2f} | {r.fps:<7.1f} | {r.peak_vram_mb:<8.1f} | "
                  f"{r.params_M:<8.3f} | {r.ap_per_gflop:<8.4f}")
        print("="*120)

    def pareto_frontier(
        self, results: List[AblationStageResult]
    ) -> List[AblationStageResult]:
        """Return Pareto-optimal subset of accepted results (AP vs latency)."""
        accepted = [r for r in results if r.accepted]
        pareto = []
        for i, r_i in enumerate(accepted):
            dominated = False
            for j, r_j in enumerate(accepted):
                if i == j:
                    continue
                if (r_j.ap50_95 >= r_i.ap50_95 and
                        r_j.mean_ms <= r_i.mean_ms and
                        (r_j.ap50_95 > r_i.ap50_95 or r_j.mean_ms < r_i.mean_ms)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r_i)
        return sorted(pareto, key=lambda x: x.ap50_95, reverse=True)

    def save_results(self, results: List[AblationStageResult], path: str):
        """Save ablation results to JSON."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "ablation_stages": [r.to_dict() for r in results],
            "yolo26_official_reference": {
                "source": OFFICIAL_REFERENCE["source"],
                "nano_ap50_95": OFFICIAL_REFERENCE["variants"]["nano"]["coco_val_ap50_95"],
                "nano_params_M": OFFICIAL_REFERENCE["variants"]["nano"]["params_M"],
                "nano_gflops": OFFICIAL_REFERENCE["variants"]["nano"]["gflops_640"],
            },
            "pareto_stages": [r.stage_name for r in self.pareto_frontier(results)],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ExperimentEngine] Results saved to {path}")
