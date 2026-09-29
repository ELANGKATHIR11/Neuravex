"""
Neuravex — Lightweight Unified Computer Vision SDK.

Core package imports without optional heavy dependencies.
Optional modules (torch, torchvision, scipy, pycocotools, etc.) are imported
lazily so that `import neuravex` succeeds on minimal environments.
"""

__version__ = "0.2.0"

# --------------------------------------------------------------------------
# Core schema (no heavy deps)
# --------------------------------------------------------------------------
from .core import ObjectState, FrameState

# --------------------------------------------------------------------------
# Lazy import helpers — heavy dependencies loaded on first access
# --------------------------------------------------------------------------
def __getattr__(name):
    """Lazy-load heavy modules on first attribute access."""
    _lazy_map = {
        # Models
        "Neuravex": (".models", "Neuravex"),
        "build_neuravex": (".models", "build_neuravex"),
        "SlotAttention": (".models.slots", "SlotAttention"),
        "UnsupervisedObjectDiscoveryHead": (".models.slots", "UnsupervisedObjectDiscoveryHead"),
        "MaskedMultimodalAutoencoder": (".models.backbone", "MaskedMultimodalAutoencoder"),
        # Engine
        "NeuravexMultiTaskTrainer": (".engine.trainer", "NeuravexMultiTaskTrainer"),
        "NeuravexInferencePostProcessor": (".engine.evaluator", "NeuravexInferencePostProcessor"),
        "RealTimeMetricDepthTracker": (".engine.tracker", "RealTimeMetricDepthTracker"),
        "robust_mask_depth_estimator": (".engine.tracker", "robust_mask_depth_estimator"),
        "calculate_map_metrics": (".engine.evaluator", "calculate_map_metrics"),
        "calculate_miou": (".engine.evaluator", "calculate_miou"),
        "calculate_depth_metrics": (".engine.evaluator", "calculate_depth_metrics"),
        "calculate_boundary_fscore": (".engine.evaluator", "calculate_boundary_fscore"),
        "calculate_bos_metrics": (".engine.evaluator", "calculate_bos_metrics"),
        "calculate_3d_iou_and_bos": (".engine.evaluator", "calculate_3d_iou_and_bos"),
        "ActorCriticComputePolicy": (".engine.adaptive_compute", "ActorCriticComputePolicy"),
        "compute_dynamic_budget_reward": (".engine.adaptive_compute", "compute_dynamic_budget_reward"),
        "RealTimeSlicedPatchSegmenter": (".engine.sliced_segmenter", "RealTimeSlicedPatchSegmenter"),
        "PrecisionPerceptionPipeline": (".engine.precision_perception", "PrecisionPerceptionPipeline"),
        # Geometry
        "CameraIntrinsics": (".geometry.camera", "CameraIntrinsics"),
        "oriented_iou_3d": (".geometry.oriented_iou3d", "oriented_iou_3d"),
        "bbox_ciou": (".geometry.box_ops", "bbox_ciou"),
        "box_iou_2d": (".geometry.box_ops", "box_iou_2d"),
        "box_giou": (".geometry.box_ops", "box_giou"),
        "box_diou": (".geometry.box_ops", "box_diou"),
        "calculate_box_overlap_score": (".geometry.box_ops", "calculate_box_overlap_score"),
        # Losses
        "RelativePoseEstimator": (".losses.photometric", "RelativePoseEstimator"),
        "ViewSynthesisWarp": (".losses.photometric", "ViewSynthesisWarp"),
        "PhotometricReconstructionLoss": (".losses.photometric", "PhotometricReconstructionLoss"),
        "PhysicsConstraintReward": (".losses.physics_rl", "PhysicsConstraintReward"),
        "PhysicsTestTimeSelfCorrector": (".losses.physics_rl", "PhysicsTestTimeSelfCorrector"),
        # Specialization
        "DatasetAnalyzer": (".specialization", "DatasetAnalyzer"),
        "ArchitectureGenerator": (".specialization", "ArchitectureGenerator"),
        "ArchSpec": (".specialization", "ArchSpec"),
        "HardwareConstraints": (".specialization", "HardwareConstraints"),
        "MultiLevelDifficultyRouter": (".specialization", "MultiLevelDifficultyRouter"),
        "HardwareProfiler": (".specialization", "HardwareProfiler"),
        "NeuravexDistillation": (".specialization", "NeuravexDistillation"),
        "ImbalanceAwareClassWeights": (".specialization", "ImbalanceAwareClassWeights"),
        "StructuredChannelPruner": (".specialization", "StructuredChannelPruner"),
        "PostTrainingQuantizer": (".specialization", "PostTrainingQuantizer"),
        "OFFICIAL_REFERENCE": (".specialization", "OFFICIAL_REFERENCE"),
        "YOLO26BaselineRunner": (".specialization", "YOLO26BaselineRunner"),
        "CompetitionResult": (".specialization", "CompetitionResult"),
        "SpecializationPipeline": (".specialization", "SpecializationPipeline"),
        "build_specialized_neuravex": (".specialization", "build_specialized_neuravex"),
        "ExperimentEngine": (".specialization", "ExperimentEngine"),
        # Hub / Model Zoo
        "download_model": (".hub", "download_model"),
        "load_model": (".hub", "load_model"),
        "list_models": (".hub", "list_models"),
        "MODEL_REGISTRY": (".hub", "MODEL_REGISTRY"),
    }

    if name in _lazy_map:
        module_path, attr_name = _lazy_map[name]
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        val = getattr(mod, attr_name)
        globals()[name] = val  # Cache for subsequent access
        return val

    raise AttributeError(f"module 'neuravex' has no attribute {name!r}")
