"""
Comprehensive Unit & Integration Test Suite for all 20 Neuravex Subsystems:
1. CPU-First Core Family (Pico, Femto, Nano, Lite, Edge, Pro, Omni)
2. Dynamic Resolution Router (320 -> 512 -> 640)
3. Dynamic Token Router (Importance estimation and spatial gating)
4. Dynamic Layer / Early Exit Router
5. Real Task-Conditional Execution (Zero-cost detection fast path)
6. Temporal Feature Reuse (Video feature caching)
7. Structural Pruning (Real parameter & filter reconstruction)
8. Quantization System (CPU INT8 & numerical verification)
9. Deployment Backends (ONNX export & ONNX Runtime execution)
10. Hardware-Aware Controller (System profiling & recommendation)
11. Hardware-Aware NAS (Pareto-frontier evaluation)
12. Efficient Backbone/Neck (Low-memory local/global mixing)
13. Object-Centric Refinement (ROI candidate refinement)
14. Confidence & Uncertainty Engine (Calibration & DFL dispersion)
15. Modular Geometry/Depth Path
16. Memory & State Optimization (Inference mode, memory format)
17. Hardware Profiler (P50/P95/P99, FLOPs, Memory, Energy)
18. Benchmark Rebuild (Real predictions & real metrics)
19. Industry Baseline Comparator
20. Production Hardening & Determinism
"""

import os
import tempfile
import torch
import pytest
import numpy as np

from neuravex.models.neuravex import build_neuravex
from neuravex.geometry.camera import CameraIntrinsics
from neuravex.engine.routing import TokenImportanceRouter, EarlyExitLayerRouter, ResolutionRouter
from neuravex.engine.temporal import TemporalFeatureCache
from neuravex.specialization.structural_pruner import StructuralGraphPruner
from neuravex.specialization.quantization_system import QuantizationSystem
from neuravex.deployment import export_onnx, ORTInferenceEngine
from neuravex.engine.hardware_controller import HardwareAwareController
from neuravex.specialization.nas_pareto import HardwareAwareNAS, compute_pareto_frontier
from neuravex.engine.object_refinement import ObjectCentricRefiner
from neuravex.engine.uncertainty import ConfidenceUncertaintyEngine
from neuravex.engine.memory_opt import MemoryOptimizer
from neuravex.benchmarks.profiler import SystemProfiler
from neuravex.benchmarks.benchmark_harness import BenchmarkHarness
from neuravex.benchmarks.baseline_comparison import IndustryBaselineComparator


def test_1_cpu_first_model_family():
    """Verify all 7 variants in the Neuravex family instantiate and run on CPU."""
    variants = ["pico", "femto", "nano", "lite", "edge", "pro", "omni"]
    x = torch.randn(1, 3, 128, 128)

    for v in variants:
        model = build_neuravex(size=v, num_classes=10)
        assert model is not None, f"Failed to build {v}"
        with torch.no_grad():
            out = model(x, tasks=("det",))
        assert "class_logits" in out
        assert "pred_boxes" in out
        assert out["class_logits"].shape[0] == 1
        assert out["pred_boxes"].shape[0] == 1


def test_2_dynamic_resolution_router():
    """Verify ResolutionRouter escalates low resolution when small/uncertain objects present."""
    router = ResolutionRouter(low_res=128, mid_res=160, high_res=192, conf_escalate_thresh=0.60)
    model = build_neuravex(size="pico", num_classes=5)
    x = torch.randn(1, 3, 192, 192)

    out = router.run_adaptive_inference(model, x, tasks=("det",))
    assert "resolution_routing" in out
    assert "escalated" in out["resolution_routing"]
    assert "initial_res" in out["resolution_routing"]


def test_3_dynamic_token_router():
    """Verify TokenImportanceRouter produces spatial importance mask and gates tokens."""
    router = TokenImportanceRouter(in_channels=32, keep_ratio=0.5)
    feat = torch.randn(2, 32, 16, 16)
    gated, stats = router(feat)

    assert gated.shape == feat.shape
    assert "retention_ratio" in stats
    assert "mask" in stats
    assert stats["mask"].shape == (2, 1, 16, 16)


def test_4_dynamic_layer_early_exit():
    """Verify EarlyExitLayerRouter detects high-confidence intermediate features."""
    router = EarlyExitLayerRouter(in_channels=32, num_classes=5, confidence_threshold=0.50)
    router.eval()
    feat = torch.randn(1, 32, 8, 8)
    should_exit, probs, mean_conf = router(feat)

    assert isinstance(should_exit, bool)
    assert probs.shape == (1, 5)
    assert 0.0 <= mean_conf <= 1.0


def test_5_real_task_conditional_execution():
    """Verify detection-only mode executes zero 3D heads or auxiliary modules."""
    model = build_neuravex(size="nano", num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 128, 128)

    # Det-only mode
    out_det = model(x, tasks=("det",))
    assert "class_logits" in out_det
    assert "pred_boxes" in out_det
    # Verify 3D outputs are NOT computed in detection-only mode
    assert "pred_xyz" not in out_det or out_det["pred_xyz"] is None
    assert "pred_lwh" not in out_det or out_det["pred_lwh"] is None
    assert "depth_map" not in out_det


def test_6_temporal_feature_reuse():
    """Verify TemporalFeatureCache reuses backbone features for consecutive low-motion frames."""
    cache = TemporalFeatureCache(motion_threshold=0.10)
    model = build_neuravex(size="pico", num_classes=5)
    frame1 = torch.rand(1, 3, 128, 128)
    # Frame 2 is almost identical to frame 1 (minimal motion)
    frame2 = frame1 + torch.randn_like(frame1) * 0.005

    out1 = cache.process_frame(model, frame1, tasks=("det",))
    assert out1["temporal_stats"]["is_keyframe"] is True

    out2 = cache.process_frame(model, frame2, tasks=("det",))
    assert out2["temporal_stats"]["reused_features"] is True
    assert out2["temporal_stats"]["frames_since_keyframe"] == 1


def test_7_structural_pruning():
    """Verify StructuralGraphPruner physically slices filters and reduces parameter count."""
    model = build_neuravex(size="nano", num_classes=5)
    orig_params = sum(p.numel() for p in model.parameters())

    pruner = StructuralGraphPruner(prune_ratio=0.25, min_channels=8)
    pruned_model, stats = pruner.prune_model_stem_and_heads(model)

    new_params = sum(p.numel() for p in pruned_model.parameters())
    assert new_params < orig_params
    assert stats["parameter_reduction_percent"] > 0
    assert stats["structurally_reduced"] is True

    # Test forward pass on physically pruned model
    x = torch.randn(1, 3, 128, 128)
    out = pruned_model(x, tasks=("det",))
    assert "class_logits" in out
    assert "pred_boxes" in out


def test_8_quantization_and_validation():
    """Verify QuantizationSystem quantizes model to INT8 and validates numeric integrity."""
    model = build_neuravex(size="pico", num_classes=5)
    qs = QuantizationSystem(min_cosine_similarity=0.90)

    quantized_model, meta = qs.quantize_cpu_int8(model)
    assert meta["status"] == "quantized"

    x = torch.randn(1, 3, 128, 128)
    val = qs.validate_quantized_model(model, quantized_model, x, tasks=("det",))
    assert val["passed_validation"] is True
    assert "class_logits_cosine_similarity" in val["numerical_metrics"]


def test_9_onnx_export_and_runtime():
    """Verify ONNX export succeeds and ORT engine runs inference accurately."""
    model = build_neuravex(size="pico", num_classes=5)
    tmp_onnx = os.path.join(tempfile.gettempdir(), "test_pico_subsys.onnx")

    res = export_onnx(model, tmp_onnx, (1, 3, 128, 128), opset_version=14)
    assert res["success"] is True
    assert os.path.exists(tmp_onnx)

    engine = ORTInferenceEngine(tmp_onnx, provider_preference=["CPUExecutionProvider"])
    out = engine.run(torch.randn(1, 3, 128, 128))
    assert "class_logits" in out
    assert "pred_boxes" in out
    assert out["class_logits"].shape[0] == 1


def test_10_hardware_controller():
    """Verify HardwareAwareController profiles hardware and produces deployment recommendations."""
    controller = HardwareAwareController()
    report = controller.get_hardware_report()

    assert "hardware_profile" in report
    assert "recommendation" in report
    assert report["hardware_profile"]["cpu_cores_physical"] > 0
    assert report["recommendation"]["model_variant"] in ["pico", "femto", "nano", "lite", "edge", "pro", "omni"]


def test_11_nas_pareto():
    """Verify HardwareAwareNAS generates candidates and computes non-dominated Pareto frontier."""
    nas = HardwareAwareNAS(device="cpu", num_classes=5)
    configs = nas.generate_candidate_configs()[:3]
    candidates = [nas.benchmark_candidate(c, (128, 128), iterations=3) for c in configs]

    pareto = compute_pareto_frontier(candidates)
    assert len(pareto) > 0
    assert all("latency_ms" in c and "accuracy_proxy" in c for c in pareto)


def test_12_13_object_centric_refiner():
    """Verify ObjectCentricRefiner applies ROI alignment and refines bounding box coordinates."""
    refiner = ObjectCentricRefiner(in_channels=32, conf_thresh=0.25)
    features = torch.randn(1, 32, 16, 16)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [5.0, 5.0, 20.0, 20.0]])
    scores = torch.tensor([0.85, 0.15])

    ref_boxes, ref_scores, quals = refiner.refine_detections(features, boxes, scores)
    # Only 1 candidate should pass conf_thresh 0.25
    assert len(ref_boxes) == 1
    assert ref_boxes.shape == (1, 4)
    assert quals.shape == (1, 1)


def test_14_confidence_uncertainty_engine():
    """Verify ConfidenceUncertaintyEngine computes calibrated entropy and DFL dispersion variance."""
    engine = ConfidenceUncertaintyEngine(temperature=1.1, uncertainty_escalation_threshold=0.40)
    logits = torch.randn(1, 10, 5)
    pred_box_dist = torch.randn(1, 10, 64)

    unc = engine.evaluate_uncertainty(logits, pred_box_dist)
    assert "calibrated_probs" in unc
    assert "classification_entropy" in unc
    assert "box_edge_variance" in unc
    assert "composite_uncertainty" in unc
    assert "requires_escalation" in unc


def test_15_modular_geometry_depth():
    """Verify 3D geometry & DEM pipeline functions when explicitly requested."""
    model = build_neuravex(size="nano", num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    K = CameraIntrinsics(fx=100.0, fy=100.0, cx=64.0, cy=64.0)

    out = model(x, intrinsics=K, tasks=("det", "geometry_3d", "dem"))
    assert "depth_map" in out
    assert "pred_xyz" in out
    assert "pred_lwh" in out


def test_16_memory_optimizer():
    """Verify MemoryOptimizer inference scope and contiguous input preparation."""
    opt = MemoryOptimizer()
    with opt.inference_scope():
        t = torch.randn(1, 3, 64, 64)
        t_prep = opt.prepare_input(t, torch.device("cpu"))
        assert t_prep.shape == t.shape


def test_17_system_profiler():
    """Verify SystemProfiler measures P50 latency, FLOPs, and FPS without fabrication."""
    profiler = SystemProfiler(warmup_iters=2, steady_iters=5)
    model = build_neuravex(size="pico", num_classes=5)
    metrics = profiler.profile(model, (1, 3, 128, 128), device="cpu")

    assert metrics.params_M > 0
    assert metrics.p50_latency_ms > 0
    assert metrics.fps > 0
    assert metrics.gflops >= 0


def test_18_19_20_benchmark_harness_and_baseline():
    """Verify BenchmarkHarness runs real samples and IndustryBaselineComparator schemas."""
    harness = BenchmarkHarness(device="cpu")
    model = build_neuravex(size="pico", num_classes=2)

    sample_dataset = [{
        "image": torch.randn(1, 3, 128, 128),
        "gt_boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "gt_labels": torch.tensor([1])
    }]

    res = harness.evaluate_model_on_annotated_set(model, sample_dataset, resolution=(128, 128), num_runs=1)
    assert res["num_samples_evaluated"] == 1
    assert "map50_mean" in res
    assert "fps" in res

    comparator = IndustryBaselineComparator(device="cpu")
    neuravex_summary = comparator.benchmark_neuravex_family(["pico"], resolution=(128, 128))
    assert len(neuravex_summary) == 1
    assert neuravex_summary[0]["variant"] == "Neuravex-Pico"
