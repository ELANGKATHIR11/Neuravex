from .assigner import TaskAlignedAssigner
from .detection_loss import DetectionLoss
from .segmentation_loss import (
    semantic_segmentation_loss, boundary_loss,
    mask_quality_loss, discriminative_instance_loss
)
from .depth_3d_loss import scale_invariant_log_depth_loss, loss_3d_detection
from .consistency_loss import TransformAlignedConsistencyLoss
from .multitask_loss import AdaptiveTaskLoss
