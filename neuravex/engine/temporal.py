"""
Temporal Feature Reuse Subsystem for Neuravex:
Reuses intermediate multi-scale backbone/neck representations between consecutive video frames.
Performs full extraction only on scene/object change events, reducing CPU/GPU latency significantly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class TemporalFeatureCache:
    """
    Temporal Feature & State Cache for Video Processing:
    1. Caches backbone multi-scale feature pyramids (P3, P4, P5).
    2. Computes inter-frame perceptual/pixel difference against the previous reference frame.
    3. Reuses cached feature maps when inter-frame delta is below motion threshold,
       performing light-weight detection/tracking updates.
    4. Automatically executes full keyframe refresh periodically or upon scene-cut detection.
    """
    def __init__(
        self,
        motion_threshold: float = 0.035,
        max_cached_frames: int = 5,
        decay_factor: float = 0.95
    ):
        self.motion_threshold = motion_threshold
        self.max_cached_frames = max_cached_frames
        self.decay_factor = decay_factor
        
        self.cached_feats: Optional[List[torch.Tensor]] = None
        self.prev_frame_gray: Optional[torch.Tensor] = None
        self.frames_since_keyframe: int = 0
        self.total_frames_processed: int = 0
        self.total_frames_reused: int = 0

    def reset(self):
        """Resets the temporal cache state."""
        self.cached_feats = None
        self.prev_frame_gray = None
        self.frames_since_keyframe = 0
        self.total_frames_processed = 0
        self.total_frames_reused = 0

    def compute_motion_delta(self, current_frame: torch.Tensor) -> float:
        """
        Computes normalized mean absolute difference between consecutive frames.
        current_frame: (B, 3, H, W) normalized image tensor
        """
        # Convert to single-channel luminance proxy for fast comparison
        # L = 0.299*R + 0.587*G + 0.114*B
        if current_frame.shape[1] == 3:
            curr_gray = (
                0.299 * current_frame[:, 0:1] +
                0.587 * current_frame[:, 1:2] +
                0.114 * current_frame[:, 2:3]
            )
        else:
            curr_gray = current_frame

        if self.prev_frame_gray is None:
            self.prev_frame_gray = curr_gray.detach().clone()
            return 1.0  # Keyframe forced

        # Compute spatial mean difference
        diff = torch.abs(curr_gray - self.prev_frame_gray).mean().item()
        self.prev_frame_gray = curr_gray.detach().clone()
        return float(diff)

    def process_frame(
        self,
        model: nn.Module,
        frame: torch.Tensor,
        tasks: tuple = ("det",),
        force_keyframe: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Processes a video frame with temporal feature reuse.
        """
        self.total_frames_processed += 1
        motion_delta = self.compute_motion_delta(frame)

        is_keyframe = (
            force_keyframe or
            self.cached_feats is None or
            self.frames_since_keyframe >= self.max_cached_frames or
            motion_delta >= self.motion_threshold
        )

        if is_keyframe:
            # Full inference: compute backbone + neck + heads
            out = model(frame, tasks=tasks, **kwargs)
            # Cache backbone/neck features if available
            p3, p4, p5 = model.backbone(frame)
            q3, q4, q5 = model.neck(p3, p4, p5)
            self.cached_feats = [q3.detach().clone(), q4.detach().clone(), q5.detach().clone()]
            self.frames_since_keyframe = 0
            reused = False
        else:
            # Reused inference: pass cached features straight through detection head
            self.frames_since_keyframe += 1
            self.total_frames_reused += 1
            reused = True
            
            # Apply slight exponential decay to cached features to avoid drift
            q3, q4, q5 = [f * self.decay_factor for f in self.cached_feats]
            
            # Evaluate detection head directly on cached features without backbone recomputation
            compute_3d = (tasks is None) or ("geometry_3d" in tasks) or ("3d" in tasks)
            det_out = model.det_head([q3, q4, q5], compute_3d=compute_3d)
            out = det_out

        out["temporal_stats"] = {
            "is_keyframe": is_keyframe,
            "reused_features": reused,
            "motion_delta": motion_delta,
            "frames_since_keyframe": self.frames_since_keyframe,
            "reuse_rate": float(self.total_frames_reused / max(1, self.total_frames_processed))
        }
        return out
