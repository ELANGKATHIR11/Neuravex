"""
neuravex.losses namespace:
Preserved for backward-compatibility, redirecting all loss implementations to neuravex.loss.
"""
from ..loss import (
    TaskAlignedAssigner, HungarianOneToOneAssigner,
    DetectionLoss, DualAssignmentDetectionLoss,
    semantic_segmentation_loss, boundary_loss,
    mask_quality_loss, discriminative_instance_loss,
    scale_invariant_log_depth_loss, loss_3d_detection,
    TransformAlignedConsistencyLoss,
    AdaptiveTaskLoss,
    RelativePoseEstimator, ViewSynthesisWarp, PhotometricReconstructionLoss,
    PhysicsConstraintReward, PhysicsTestTimeSelfCorrector
)

