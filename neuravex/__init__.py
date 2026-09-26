from .models import Neuravex, build_neuravex
from .models.slots import SlotAttention, UnsupervisedObjectDiscoveryHead
from .models.backbone import MaskedMultimodalAutoencoder
from .losses import (
    RelativePoseEstimator,
    ViewSynthesisWarp,
    PhotometricReconstructionLoss,
    PhysicsConstraintReward,
    PhysicsTestTimeSelfCorrector
)
from .engine import (
    NeuravexMultiTaskTrainer,
    NeuravexInferencePostProcessor,
    RealTimeMetricDepthTracker,
    robust_mask_depth_estimator,
    calculate_map_metrics,
    calculate_miou,
    calculate_depth_metrics,
    calculate_boundary_fscore,
    calculate_bos_metrics,
    calculate_3d_iou_and_bos,
    ActorCriticComputePolicy,
    compute_dynamic_budget_reward,
    RealTimeSlicedPatchSegmenter
)
from .geometry import (
    CameraIntrinsics,
    oriented_iou_3d,
    bbox_ciou,
    box_iou_2d,
    box_giou,
    box_diou,
    calculate_box_overlap_score
)



from .specialization import (
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
)

__version__ = "0.1.0"
