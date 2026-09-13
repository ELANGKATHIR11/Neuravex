"""
neuravex.specialization — Dataset-Conditioned + Hardware-Aware + Adaptive-Compute
Specialization Paradigm for Neuravex.

Core transform:
  user dataset + task + hardware + constraints
  → DatasetAnalyzer (complexity vector)
  → ArchitectureGenerator (ArchSpec)
  → build_specialized_neuravex (model)
  → MultiLevelDifficultyRouter (easy/medium/hard routing)
  → HardwareProfiler (real latency)
  → NeuravexDistillation (optional KD)
  → StructuredChannelPruner + PostTrainingQuantizer (optional compression)
  → ExperimentEngine (controlled ablations)
  → YOLO26BaselineRunner (competition gate)
"""

from .dataset_analyzer import DatasetAnalyzer, DatasetStats
from .architecture_generator import ArchitectureGenerator, ArchSpec, HardwareConstraints
from .difficulty_router import (
    MultiLevelDifficultyRouter,
    DifficultyScorer,
    LightRefinementBlock,
    FullRefinementBlock,
    RoutingMetricsLogger,
)
from .hardware_profiler import HardwareProfiler, HardwareProfile, count_parameters, estimate_gflops
from .distillation import (
    NeuravexDistillation,
    LogitDistillationLoss,
    BoxDistributionDistillationLoss,
    FeatureHintLoss,
    ImbalanceAwareClassWeights,
)
from .pruning_quantization import (
    StructuredChannelPruner,
    PostTrainingQuantizer,
    PruningResult,
    QuantizationResult,
)
from .yolo26_baseline import (
    OFFICIAL_REFERENCE,
    YOLO26BaselineRunner,
    YOLO26LocalReproResult,
    CompetitionResult,
)
from .specialization_pipeline import (
    SpecializationPipeline,
    SpecializationReport,
    build_specialized_neuravex,
)
from .experiment_engine import ExperimentEngine, AblationStageResult

__all__ = [
    # Dataset Analysis
    "DatasetAnalyzer",
    "DatasetStats",
    # Architecture Generation
    "ArchitectureGenerator",
    "ArchSpec",
    "HardwareConstraints",
    # Difficulty Routing
    "MultiLevelDifficultyRouter",
    "DifficultyScorer",
    "LightRefinementBlock",
    "FullRefinementBlock",
    "RoutingMetricsLogger",
    # Hardware Profiling
    "HardwareProfiler",
    "HardwareProfile",
    "count_parameters",
    "estimate_gflops",
    # Distillation
    "NeuravexDistillation",
    "LogitDistillationLoss",
    "BoxDistributionDistillationLoss",
    "FeatureHintLoss",
    "ImbalanceAwareClassWeights",
    # Pruning/Quantization
    "StructuredChannelPruner",
    "PostTrainingQuantizer",
    "PruningResult",
    "QuantizationResult",
    # YOLO26 Baseline
    "OFFICIAL_REFERENCE",
    "YOLO26BaselineRunner",
    "YOLO26LocalReproResult",
    "CompetitionResult",
    # Pipeline
    "SpecializationPipeline",
    "SpecializationReport",
    "build_specialized_neuravex",
    # Experiment Engine
    "ExperimentEngine",
    "AblationStageResult",
]
