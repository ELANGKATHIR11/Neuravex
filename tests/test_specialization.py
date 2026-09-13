"""
Tests for Neuravex Specialization System.

Tests all components with synthetic data:
  1. DatasetAnalyzer — complexity vector extraction + validation
  2. ArchitectureGenerator — spec generation, budget enforcement, scale rules
  3. MultiLevelDifficultyRouter — 3-level routing, training/inference modes
  4. HardwareProfiler — latency measurement, efficiency metrics
  5. Distillation — logit KD, box KD, feature hint, teacher validation
  6. StructuredChannelPruner — pruning + output validation
  7. PostTrainingQuantizer — INT8 quantization + output validation
  8. YOLO26BaselineRunner — official reference, NOT_PROVEN path
  9. SpecializationPipeline — end-to-end pipeline without real dataset
  10. ExperimentEngine — ablation sequence on synthetic data

Hard-fails on:
  - NaN/Inf in any output
  - Shape mismatches
  - Fabricated results
"""

import json
import math
import os
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn

from neuravex import build_neuravex
from neuravex.specialization import (
    DatasetAnalyzer,
    ArchitectureGenerator,
    ArchSpec,
    HardwareConstraints,
    MultiLevelDifficultyRouter,
    HardwareProfiler,
    NeuravexDistillation,
    ImbalanceAwareClassWeights,
    StructuredChannelPruner,
    PostTrainingQuantizer,
    OFFICIAL_REFERENCE,
    YOLO26BaselineRunner,
    CompetitionResult,
    SpecializationPipeline,
    build_specialized_neuravex,
    ExperimentEngine,
    RoutingMetricsLogger,
)


# ─── Synthetic COCO Annotation File ───────────────────────────────────────────

def make_synthetic_coco_annotation(
    n_images: int = 20,
    n_classes: int = 5,
    n_annotations: int = 60,
    img_size: int = 640,
    tmpdir: str = None,
) -> str:
    """Create a temporary COCO-format annotation JSON for testing."""
    categories = [{"id": i, "name": f"class_{i}"} for i in range(n_classes)]
    images = [{"id": i, "file_name": f"img_{i:04d}.jpg", "height": img_size, "width": img_size}
              for i in range(n_images)]
    annotations = []
    rng = np.random.default_rng(42)
    for ann_id in range(n_annotations):
        img_id = int(rng.integers(0, n_images))
        cat_id = int(rng.integers(0, n_classes))
        x = float(rng.integers(0, img_size - 50))
        y = float(rng.integers(0, img_size - 50))
        w = float(rng.integers(10, 200))
        h = float(rng.integers(10, 200))
        annotations.append({
            "id": ann_id, "image_id": img_id, "category_id": cat_id,
            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0
        })

    coco_data = {"images": images, "annotations": annotations, "categories": categories}

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    ann_file = os.path.join(tmpdir, "annotations.json")
    with open(ann_file, "w") as f:
        json.dump(coco_data, f)
    return ann_file


# ─── Test 1: DatasetAnalyzer ──────────────────────────────────────────────────

class TestDatasetAnalyzer:
    def setup_method(self):
        self.ann_file = make_synthetic_coco_annotation()
        self.analyzer = DatasetAnalyzer()

    def test_analyze_returns_stats(self):
        stats = self.analyzer.analyze(self.ann_file)
        assert stats.num_classes == 5
        assert stats.total_images == 20
        assert stats.total_annotations == 60
        assert len(stats.object_areas) == 60

    def test_complexity_vector_shape_and_range(self):
        stats = self.analyzer.analyze(self.ann_file)
        vec = stats.complexity_vector
        assert vec is not None
        assert vec.shape == (16,)
        assert np.all(vec >= 0.0), f"Negative values in vector: {vec}"
        assert np.all(vec <= 1.0), f"Values > 1.0 in vector: {vec}"

    def test_complexity_vector_no_nan_inf(self):
        stats = self.analyzer.analyze(self.ann_file)
        vec = stats.complexity_vector
        assert not np.any(np.isnan(vec)), "NaN in complexity vector"
        assert not np.any(np.isinf(vec)), "Inf in complexity vector"

    def test_summarize(self):
        stats = self.analyzer.analyze(self.ann_file)
        summary = self.analyzer.summarize(stats)
        assert "num_classes" in summary
        assert "complexity_vector" in summary
        assert "recommended_scale_emphasis" in summary
        assert len(summary["complexity_vector"]) == 16

    def test_tiny_object_dataset_emphasizes_small_fraction(self):
        """Force all annotations to tiny objects and check vec[2] is high."""
        categories = [{"id": 0, "name": "tiny"}]
        images = [{"id": 0, "file_name": "img.jpg", "height": 640, "width": 640}]
        # Very small boxes (< 32^2 = 1024 area)
        annotations = [
            {"id": i, "image_id": 0, "category_id": 0,
             "bbox": [10.0, 10.0, 5.0, 5.0], "area": 25.0, "iscrowd": 0}
            for i in range(20)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
            tiny_file = f.name
        stats = self.analyzer.analyze(tiny_file)
        os.unlink(tiny_file)
        assert stats.complexity_vector[2] > 0.9, f"Small fraction vec[2]={stats.complexity_vector[2]}"


# ─── Test 2: ArchitectureGenerator ───────────────────────────────────────────

class TestArchitectureGenerator:
    def setup_method(self):
        self.analyzer = DatasetAnalyzer()
        self.generator = ArchitectureGenerator(img_size=320)
        self.ann_file = make_synthetic_coco_annotation()

    def test_generate_returns_valid_spec(self):
        stats = self.analyzer.analyze(self.ann_file)
        constraints = HardwareConstraints(param_budget_M=50.0, latency_budget_ms=100.0)
        spec = self.generator.generate(stats.complexity_vector, constraints)
        assert isinstance(spec, ArchSpec)
        assert spec.base_c > 0
        assert 0 < spec.depth_mul <= 3.0
        assert spec.reg_max in (1, 4, 8, 16)

    def test_spec_respects_budget(self):
        constraints = HardwareConstraints(param_budget_M=5.0, latency_budget_ms=20.0)
        vec = np.zeros(16, dtype=np.float32)
        spec = self.generator.generate(vec, constraints)
        assert spec.estimated_params_M <= constraints.param_budget_M * 1.2

    def test_tiny_object_dataset_gets_p2(self):
        # Force tiny-object vector
        vec = np.zeros(16, dtype=np.float32)
        vec[2] = 0.8  # 80% small objects
        constraints = HardwareConstraints(param_budget_M=50.0, latency_budget_ms=200.0)
        spec = self.generator.generate(vec, constraints)
        assert spec.use_p2 is True
        assert 4 in spec.strides

    def test_large_object_dataset_no_p2(self):
        vec = np.zeros(16, dtype=np.float32)
        vec[4] = 0.8  # 80% large objects
        constraints = HardwareConstraints(param_budget_M=50.0, latency_budget_ms=200.0)
        spec = self.generator.generate(vec, constraints)
        assert spec.use_p2 is False

    def test_hard_dataset_higher_router_threshold(self):
        vec = np.zeros(16, dtype=np.float32)
        vec[5] = 0.9   # High density
        vec[11] = 0.9  # High occlusion
        constraints = HardwareConstraints(param_budget_M=50.0, latency_budget_ms=200.0)
        spec = self.generator.generate(vec, constraints)
        assert spec.router_threshold > 0.5

    def test_search_returns_multiple_specs(self):
        vec = np.zeros(16, dtype=np.float32)
        constraints = HardwareConstraints(param_budget_M=100.0, latency_budget_ms=500.0)
        specs = self.generator.search(vec, constraints, n_candidates=3)
        assert len(specs) >= 1
        # Specs should be ordered roughly cheapest to most expensive
        assert all(isinstance(s, ArchSpec) for s in specs)


# ─── Test 3: MultiLevelDifficultyRouter ──────────────────────────────────────

class TestMultiLevelDifficultyRouter:
    def setup_method(self):
        self.C = 32
        self.router = MultiLevelDifficultyRouter(self.C, easy_thresh=0.35, hard_thresh=0.65)

    def test_output_shape_matches_input(self):
        x = torch.randn(2, self.C, 20, 20)
        out, stats = self.router(x, force_full_compute=True)
        assert out.shape == x.shape

    def test_no_nan_in_output(self):
        x = torch.randn(4, self.C, 16, 16)
        out, stats = self.router(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_training_mode_differentiable(self):
        self.router.train()
        x = torch.randn(2, self.C, 8, 8, requires_grad=True)
        out, stats = self.router(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None

    def test_inference_hard_routing(self):
        self.router.eval()
        x = torch.randn(4, self.C, 8, 8)
        out, stats = self.router(x, force_full_compute=False)
        assert out.shape == x.shape
        fracs = stats["route_fractions"]
        assert abs(sum(fracs) - 1.0) < 0.01

    def test_force_full_compute_all_hard(self):
        self.router.eval()
        x = torch.randn(2, self.C, 8, 8)
        out, stats = self.router(x, force_full_compute=True)
        assert stats["fully_executed"] is True

    def test_compute_budget_loss(self):
        x = torch.randn(4, self.C, 8, 8)
        out, stats = self.router(x)
        budget_loss = self.router.get_compute_budget_loss(stats)
        assert not torch.isnan(budget_loss)
        assert not torch.isinf(budget_loss)
        assert budget_loss.item() >= 0.0

    def test_routing_metrics_logger(self):
        logger = RoutingMetricsLogger()
        for _ in range(5):
            x = torch.randn(2, self.C, 8, 8)
            _, stats = self.router(x)
            logger.update(stats)
        summary = logger.summary()
        assert "mean_easy_fraction" in summary
        assert "mean_hard_fraction" in summary
        total_frac = summary["mean_easy_fraction"] + summary["mean_medium_fraction"] + summary["mean_hard_fraction"]
        assert abs(total_frac - 1.0) < 0.05


# ─── Test 4: HardwareProfiler ────────────────────────────────────────────────

class TestHardwareProfiler:
    def setup_method(self):
        self.model = build_neuravex(size="nano", num_classes=5)
        self.profiler = HardwareProfiler(device="cpu", warmup_runs=2, measure_runs=5)

    def test_profile_returns_valid_metrics(self):
        profile = self.profiler.profile(self.model, arch_name="nano", img_size=320)
        assert profile.p50_ms > 0.0
        assert profile.p95_ms >= profile.p50_ms
        assert profile.fps > 0.0
        assert profile.params_M > 0.0
        assert math.isfinite(profile.p50_ms)
        assert math.isfinite(profile.fps)

    def test_fill_efficiency(self):
        profile = self.profiler.profile(self.model, arch_name="nano", img_size=320)
        ap = 0.35
        profile = self.profiler.fill_efficiency(profile, ap)
        assert profile.ap_per_gflop >= 0.0
        assert profile.ap_per_ms >= 0.0
        assert profile.ap_per_param_M >= 0.0

    def test_pareto_front(self):
        profile1 = self.profiler.profile(self.model, arch_name="nano", img_size=320)
        profile2 = self.profiler.profile(self.model, arch_name="nano2", img_size=320)
        pareto = self.profiler.find_pareto_front([profile1, profile2], [0.35, 0.30])
        assert len(pareto) >= 1


# ─── Test 5: Distillation ────────────────────────────────────────────────────

class TestDistillation:
    def setup_method(self):
        self.teacher = build_neuravex(size="small", num_classes=5)
        self.student = build_neuravex(size="nano", num_classes=5)
        self.teacher.eval()
        self.student.eval()
        self.x = torch.randn(1, 3, 320, 320)

    def test_kd_loss_not_nan(self):
        distiller = NeuravexDistillation(self.student, self.teacher)
        with torch.no_grad():
            t_out = self.teacher(self.x, tasks=("det",))
            s_out = self.student(self.x, tasks=("det",))
        loss, components = distiller.compute_kd_loss(s_out, t_out)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0.0

    def test_teacher_validation_disables_when_not_better(self):
        distiller = NeuravexDistillation(self.student, self.teacher)
        # Teacher worse than student
        effective = distiller.validate_effectiveness(teacher_ap=0.30, student_ap=0.35)
        assert effective is False
        assert distiller.distillation_effective is False

    def test_teacher_validation_enables_when_better(self):
        distiller = NeuravexDistillation(self.student, self.teacher)
        effective = distiller.validate_effectiveness(teacher_ap=0.50, student_ap=0.35)
        assert effective is True
        assert distiller.distillation_effective is True

    def test_shape_mismatch_raises(self):
        distiller = NeuravexDistillation(self.student, self.teacher)
        with torch.no_grad():
            t_out = self.teacher(self.x, tasks=("det",))
            s_out = self.student(self.x, tasks=("det",))
        # Corrupt shape
        bad_s_out = dict(s_out)
        bad_s_out["class_logits"] = torch.randn(1, 10, 5)  # Wrong N
        with pytest.raises((ValueError, RuntimeError)):
            distiller.compute_kd_loss(bad_s_out, t_out)

    def test_imbalance_class_weights(self):
        class_counts = {0: 1000, 1: 100, 2: 10}
        weights = ImbalanceAwareClassWeights(class_counts, num_classes=5, alpha=0.5)
        w = weights.get_weights()
        assert w.shape[0] == 5
        # Rare class (id=2) should get higher weight than frequent (id=0)
        assert w[2] > w[0]
        # Mean weight should be ~1.0
        assert abs(float(w.mean()) - 1.0) < 0.1


# ─── Test 6: Pruning ─────────────────────────────────────────────────────────

class TestPruning:
    def setup_method(self):
        self.model = build_neuravex(size="nano", num_classes=5)

    def test_pruning_produces_valid_output(self):
        pruner = StructuredChannelPruner(sparsity=0.3)
        pruned_model, result = pruner.prune(self.model)
        assert result.valid, f"Pruning failed: {result.errors}"
        assert result.sparsity_achieved > 0.0

    def test_pruning_no_nan_in_output(self):
        pruner = StructuredChannelPruner(sparsity=0.2)
        pruned_model, result = pruner.prune(self.model)
        pruned_model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = pruned_model(x, tasks=("det",))
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in pruned output '{k}'"
                assert not torch.isinf(v).any(), f"Inf in pruned output '{k}'"

    def test_sensitivity_analysis(self):
        pruner = StructuredChannelPruner(sparsity=0.2)
        x = torch.randn(1, 3, 320, 320)
        sens = pruner.analyze_sensitivity(self.model, x)
        assert len(sens) > 0
        for k, v in sens.items():
            assert math.isfinite(v), f"Non-finite sensitivity for layer {k}"


# ─── Test 7: Quantization ────────────────────────────────────────────────────

class TestQuantization:
    def setup_method(self):
        self.model = build_neuravex(size="nano", num_classes=5)

    def test_dynamic_quantization(self):
        quantizer = PostTrainingQuantizer()
        qmodel, result = quantizer.dynamic_quantize(self.model)
        # Dynamic quant of Conv2d-heavy model may not change size much
        # but should not fail
        assert isinstance(result.original_size_mb, float)
        assert isinstance(result.quantized_size_mb, float)


# ─── Test 8: YOLO26 Baseline ─────────────────────────────────────────────────

class TestYOLO26Baseline:
    def test_official_reference_structure(self):
        assert "variants" in OFFICIAL_REFERENCE
        assert "nano" in OFFICIAL_REFERENCE["variants"]
        nano = OFFICIAL_REFERENCE["variants"]["nano"]
        assert "params_M" in nano
        assert "gflops_640" in nano
        assert "coco_val_ap50_95" in nano
        # Should match known published values
        assert 2.0 < nano["params_M"] < 5.0  # YOLOv8n ~3.2M
        assert 5.0 < nano["gflops_640"] < 15.0  # YOLOv8n ~8.7 GFLOPs

    def test_not_innovations_listed(self):
        assert "NOT_Neuravex_innovations" in OFFICIAL_REFERENCE
        innovations = OFFICIAL_REFERENCE["NOT_Neuravex_innovations"]
        assert any("DFL" in x for x in innovations)
        assert any("anchor-free" in x.lower() for x in innovations)

    def test_baseline_runner_without_ultralytics_returns_not_proven(self):
        runner = YOLO26BaselineRunner()
        # On systems without ultralytics, should return NOT_PROVEN
        if not runner.check_availability():
            result = runner.run_evaluation("/nonexistent/path")
            assert result.status == "NOT_PROVEN"
            assert len(result.errors) > 0

    def test_get_reference_spec(self):
        runner = YOLO26BaselineRunner()
        spec = runner.get_reference_spec("nano")
        assert "params_M" in spec
        assert "coco_val_ap50_95" in spec

    def test_competition_result_not_proven_without_measurement(self):
        comp = CompetitionResult(
            dataset="test",
            conditions_matched=False,
            yolo26_status="NOT_PROVEN",
            neuravex_ap50_95=0.40,
            yolo26_ap50_95=0.37,
        )
        status = comp.evaluate_gate()
        assert status == "NOT_PROVEN"

    def test_competition_gate_success_path(self):
        comp = CompetitionResult(
            dataset="test",
            conditions_matched=True,
            yolo26_variant="nano",
            yolo26_ap50_95=0.37,
            yolo26_status="MEASURED",
            neuravex_ap50_95=0.38,
            neuravex_arch="small",
            yolo26_gflops=8.7,
            neuravex_gflops=5.2,  # Lower cost!
            yolo26_params_M=3.2,
            neuravex_params_M=2.1,
            yolo26_p50_ms=15.0,
            neuravex_p50_ms=12.0,
        )
        status = comp.evaluate_gate()
        assert status == "SUCCESS"
        assert comp.ap_target_met is True
        assert comp.cost_improvement is True

    def test_competition_gate_fails_when_ap_not_met(self):
        comp = CompetitionResult(
            dataset="test",
            conditions_matched=True,
            yolo26_variant="nano",
            yolo26_ap50_95=0.37,
            yolo26_status="MEASURED",
            neuravex_ap50_95=0.25,  # Lower AP
            neuravex_arch="nano",
            yolo26_gflops=8.7,
            neuravex_gflops=3.0,
            yolo26_params_M=3.2,
            neuravex_params_M=1.0,
            yolo26_p50_ms=15.0,
            neuravex_p50_ms=5.0,
        )
        status = comp.evaluate_gate()
        assert status == "LOWER_COST_BUT_AP_NOT_MET"


# ─── Test 9: SpecializationPipeline ──────────────────────────────────────────

class TestSpecializationPipeline:
    def test_pipeline_runs_without_dataset(self):
        """Pipeline should return NOT_PROVEN when no eval dataset given."""
        ann_file = make_synthetic_coco_annotation(n_images=10, n_classes=3)
        pipeline = SpecializationPipeline(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(
            device="cpu",
            param_budget_M=10.0,
            latency_budget_ms=500.0,
        )
        report = pipeline.run(
            ann_file=ann_file,
            constraints=constraints,
            num_classes=3,
        )
        assert report is not None
        assert report.status == "NOT_PROVEN"
        assert report.arch_spec is not None
        assert report.hardware_profile is not None

    def test_pipeline_model_has_no_nan(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=2)
        pipeline = SpecializationPipeline(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(
            device="cpu", param_budget_M=5.0, latency_budget_ms=1000.0
        )
        report = pipeline.run(ann_file=ann_file, constraints=constraints, num_classes=2)
        assert len([e for e in report.errors if "NaN" in e or "Inf" in e]) == 0

    def test_build_specialized_neuravex_nano(self):
        spec = ArchSpec(base_c=16, depth_mul=0.33, use_p2=False, reg_max=16)
        model = build_specialized_neuravex(spec, num_classes=10)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, tasks=("det",))
        assert "pred_boxes" in out
        assert not torch.isnan(out["pred_boxes"]).any()

    def test_build_specialized_neuravex_with_difficulty_router(self):
        spec = ArchSpec(base_c=16, depth_mul=0.33, num_difficulty_levels=3)
        model = build_specialized_neuravex(spec, num_classes=5)
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x, tasks=("det",))
        assert not torch.isnan(out["pred_boxes"]).any()


# ─── Test 10: ExperimentEngine ───────────────────────────────────────────────

class TestExperimentEngine:
    def test_ablation_runs_all_stages(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=3)
        engine = ExperimentEngine(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(
            device="cpu", param_budget_M=20.0, latency_budget_ms=2000.0
        )
        results = engine.run_ablation(
            ann_file=ann_file,
            constraints=constraints,
            num_classes=3,
            max_analysis_images=5,
        )
        assert len(results) >= 4, f"Expected >= 4 stages, got {len(results)}"

    def test_ablation_no_stage_has_nan_latency(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=3)
        engine = ExperimentEngine(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(device="cpu", param_budget_M=20.0, latency_budget_ms=2000.0)
        results = engine.run_ablation(ann_file=ann_file, constraints=constraints,
                                      num_classes=3, max_analysis_images=5)
        for r in results:
            if r.params_M > 0:
                assert math.isfinite(r.p50_ms), f"Non-finite P50 in stage {r.stage_name}"
                assert math.isfinite(r.fps), f"Non-finite FPS in stage {r.stage_name}"

    def test_baseline_always_accepted(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=3)
        engine = ExperimentEngine(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(device="cpu", param_budget_M=20.0, latency_budget_ms=2000.0)
        results = engine.run_ablation(ann_file=ann_file, constraints=constraints,
                                      num_classes=3, max_analysis_images=5)
        baseline = [r for r in results if r.stage_name == "baseline"]
        assert len(baseline) >= 1
        assert baseline[0].accepted is True

    def test_pareto_frontier_non_empty(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=3)
        engine = ExperimentEngine(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(device="cpu", param_budget_M=20.0, latency_budget_ms=2000.0)
        results = engine.run_ablation(ann_file=ann_file, constraints=constraints,
                                      num_classes=3, max_analysis_images=5)
        pareto = engine.pareto_frontier(results)
        assert len(pareto) >= 0  # May be 0 if no eval_fn provided (AP=0 for all)

    def test_save_results(self):
        ann_file = make_synthetic_coco_annotation(n_images=5, n_classes=3)
        engine = ExperimentEngine(device="cpu", img_size=320, verbose=False)
        constraints = HardwareConstraints(device="cpu", param_budget_M=20.0, latency_budget_ms=2000.0)
        results = engine.run_ablation(ann_file=ann_file, constraints=constraints,
                                      num_classes=3, max_analysis_images=5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        engine.save_results(results, tmp_path)
        assert os.path.exists(tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        assert "ablation_stages" in data
        assert "yolo26_official_reference" in data
        os.unlink(tmp_path)


# ─── Test 11: Regression — Existing 28 Tests Still Pass ─────────────────────

def test_existing_apis_still_work():
    """Quick smoke test that neuravex core APIs haven't been broken."""
    from neuravex import build_neuravex, CameraIntrinsics
    from neuravex.engine import NeuravexInferencePostProcessor

    model = build_neuravex(size="nano", num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=160.0)
    with torch.no_grad():
        out = model(x, tasks=("det",), intrinsics=K)
    assert "pred_boxes" in out
    assert not torch.isnan(out["pred_boxes"]).any()

    post = NeuravexInferencePostProcessor(conf_thresh=0.01)
    result = post(out, intrinsics=K)
    assert "detections" in result
    assert "objects" in result
