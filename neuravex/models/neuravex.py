import torch
import torch.nn as nn
from .backbone import Backbone
from .neck import PANetNeck, BidirectionalCrossTaskFusion
from .heads_det import MultiScaleDetectionHead
from .heads_seg import MultiLayerSegmentationHead
from .heads_depth import CameraAwareDEM
from .heads_pose import DepthAwareGeometryPoseHead
from .heads_st_intelligence import SpatioTemporalIntelligenceHead
from .heads_flow_counter import PromptablePrototypeHead
from ..geometry.camera import CameraIntrinsics

class Neuravex(nn.Module):
    """
    Neuravex — Lightweight Custom Computer Vision Architecture.
    Unifying 2D/3D, Dense Multitask, Human Pose, and Spatio-Temporal Video Intelligence:
      1. 2D Multi-scale anchor-free detection (P3, P4, P5 at strides 8, 16, 32) with DFL
      2. 3D detection: (X, Y, Z), (L, W, H), yaw (sin theta, cos theta) with pinhole geometry
      3. Semantic segmentation (multiclass raw logits)
      4. Boundary segmentation (raw logits for BCE/Dice)
      5. Instance discriminative embeddings + prototype masks
      6. Mask quality prediction (continuous IoU supervision)
      7. Dense inverse depth & metric depth (DEM)
      8. True bidirectional cross-task feature fusion with dedicated task branches
      9. Depth-Aware Geometry-Gated Human Pose Estimation (17 COCO 2D/3D joints)
      10. Neural Spatio-Temporal Motion, Spatial Interaction & Anomaly Reasoning
      11. SAM-Inspired Promptable Few-Shot Prototype Distillation for Industrial Flow Counting
    """
    def __init__(self, num_classes: int = 80, base_c: int = 48, depth_mul: float = 1.0, seg_embed: int = 16, num_parts: int = 16, reg_max: int = 16, enable_slots: bool = False, num_slots: int = 8, enable_pose: bool = True, enable_st_intel: bool = True, enable_prompt: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.base_c = base_c
        self.depth_mul = depth_mul
        self.reg_max = reg_max
        self.enable_slots = enable_slots
        self.enable_pose = enable_pose
        self.enable_st_intel = enable_st_intel
        self.enable_prompt = enable_prompt
        neck_c = base_c * 4

        # Backbone & FPN Neck
        self.backbone = Backbone(base_c=base_c, depth_mul=depth_mul)
        self.neck = PANetNeck(base_c=base_c)

        # Cross-Task Fusion at multi-scale
        self.fusion_p3 = BidirectionalCrossTaskFusion(neck_c)

        # Task Heads
        self.det_head = MultiScaleDetectionHead(in_channels=neck_c, num_classes=num_classes, reg_max=reg_max)
        self.seg_head = MultiLayerSegmentationHead(in_channels=neck_c, num_classes=num_classes, embed_dim=seg_embed, num_parts=num_parts)
        self.dem_head = CameraAwareDEM(in_channels=neck_c)
        self.pose_head = DepthAwareGeometryPoseHead(in_channels=neck_c) if enable_pose else None
        self.st_head = SpatioTemporalIntelligenceHead(token_dim=64, hidden_dim=128) if enable_st_intel else None
        self.prompt_head = PromptablePrototypeHead(in_channels=neck_c) if enable_prompt else None

        # Unsupervised Object Discovery (Slot Attention)
        if enable_slots:
            from .slots import UnsupervisedObjectDiscoveryHead
            self.slots_head = UnsupervisedObjectDiscoveryHead(in_channels=neck_c, num_slots=num_slots)
        else:
            self.slots_head = None

        # Adaptive compute router for high-throughput detection
        from ..engine.adaptive_compute import AdaptiveComputeRouter, ActorCriticComputePolicy
        self.router_p3 = AdaptiveComputeRouter(neck_c)
        self.rl_policy = ActorCriticComputePolicy(state_dim=neck_c)

    def forward(self, x: torch.Tensor, intrinsics: CameraIntrinsics = None, tasks: tuple = None,
                routing_threshold: float = 0.5, force_full_compute: bool = False,
                use_rl_policy: bool = False, trajectory_tokens: torch.Tensor = None,
                pairwise_indices: torch.Tensor = None, prompt_prototype: torch.Tensor = None) -> dict:
        """
        tasks: tuple of active tasks. If None, runs all configured modalities.
        For detection-only inference: pass tasks=('det',), which bypasses seg and DEM heads
        saving significant FLOPs and latency!
        routing_threshold: confidence threshold for the adaptive router [0, 1].
        force_full_compute: if True, deterministically executes all refinement blocks.
        use_rl_policy: if True, uses RL Actor-Critic policy to govern computation.
        """
        B, _, H, W = x.shape
        out_hw = (H, W)

        # 1. Multi-scale feature extraction
        p3, p4, p5 = self.backbone(x)
        q3, q4, q5 = self.neck(p3, p4, p5)

        outputs = {}

        explicit_tasks = tasks
        # Optional RL Compute Policy action
        if use_rl_policy:
            action, log_prob, val, ent = self.rl_policy(q3)
            outputs["rl_action"] = action
            outputs["rl_log_prob"] = log_prob
            outputs["rl_value"] = val
            # Action 0: Light det only; Action 1: Standard det+DEM; Action 2: Full dense multitask
            if explicit_tasks is None:
                if action.max() == 0:
                    tasks = ("det",)
                elif action.max() == 1:
                    tasks = ("det", "dem")

        # 2. Bidirectional Cross-Task Fusion across task tokens
        if tasks is not None and len(tasks) == 1 and ("det" in tasks or "geometry_3d" in tasks):
            q3_det = q3
            q3_seg = q3
            q3_dep = q3
        else:
            q3_det, q3_seg, q3_dep = self.fusion_p3(q3, q3, q3)

        # 3. Dense Elevation / Metric Depth Model (CameraAwareDEM)
        dem_depth_map = None
        if tasks is None or "depth" in tasks or "dssl" in tasks or "dem" in tasks or "terrain" in tasks:
            depth_out = self.dem_head(q3_dep, q4, q5, out_hw, intrinsics=intrinsics)
            outputs.update(depth_out)
            dem_depth_map = depth_out.get("depth_map", None)

        # 4. Multi-scale detection and 3D prediction with DEM Cross-Gating & Dual-Head
        if tasks is None or "det" in tasks or "geometry_3d" in tasks:
            # Adaptive Compute Router connected directly into Q3 detection stream
            q3_det, routing_stats = self.router_p3(
                q3_det, threshold=routing_threshold, force_full_compute=force_full_compute
            )
            outputs["routing_stats"] = routing_stats

            det_feats = [q3_det, q4, q5]
            det_out = self.det_head(det_feats, intrinsics=intrinsics, dem_depth_map=dem_depth_map)
            outputs.update(det_out)

        # 5. Multi-layer segmentation
        if tasks is None or "semantic" in tasks or "instance" in tasks or "boundary" in tasks:
            seg_out = self.seg_head(q3_seg, q4, q5, out_hw)
            outputs.update(seg_out)

        # 6. Depth-Aware Geometry-Gated Human Pose Estimation
        if self.pose_head is not None and (tasks is None or "pose" in tasks or "human" in tasks or "skeleton" in tasks):
            pose_out = self.pose_head([q3_seg, q4, q5], intrinsics=intrinsics, dem_depth_map=dem_depth_map)
            outputs.update(pose_out)

        # 7. Unsupervised Object Discovery (Slot Attention)
        if self.slots_head is not None and (explicit_tasks is None or "slots" in explicit_tasks or "discovery" in explicit_tasks):
            slots_out = self.slots_head(q5)
            outputs.update(slots_out)

        # 8. Neural Spatio-Temporal Motion, Interaction & Anomaly Reasoning
        if trajectory_tokens is not None and self.st_head is not None:
            st_out = self.st_head(trajectory_tokens, pairwise_indices=pairwise_indices)
            outputs.update(st_out)

        # 9. SAM-Inspired Promptable Prototype Distillation (Few-Shot Flow Discovery)
        if self.prompt_head is not None and (prompt_prototype is not None or (tasks is not None and "prompt" in tasks)):
            prompt_out = self.prompt_head(q3_seg, prototype=prompt_prototype)
            outputs.update(prompt_out)

        return outputs




    def switch_to_deploy(self):
        """Switches all reparameterizable blocks in the backbone to fused 3x3 convolutions."""
        for m in self.modules():
            if m is not self and hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()

def build_neuravex(size: str = "medium", num_classes: int = 80, reg_max: int = 16, enable_pose: bool = True, enable_st_intel: bool = True, enable_prompt: bool = True) -> Neuravex:
    """
    Factory function for scalable Neuravex variants spanning IoT/CPU to Data-Center GPUs:
      pico   : base_c = 8,  depth_mul = 0.25 (Ultra-low compute for microcontrollers/embedded zero-GPU IoT)
      femto  : base_c = 12, depth_mul = 0.33 (Low-power edge CPU / Raspberry Pi)
      nano   : base_c = 16, depth_mul = 0.33 (Standard CPU & mobile real-time)
      micro  : base_c = 24, depth_mul = 0.50 (Entry-level embedded GPUs like Jetson Nano)
      small  : base_c = 32, depth_mul = 0.67 (Sub-50W edge GPUs like Jetson Orin Nano / RTX 3050)
      medium : base_c = 48, depth_mul = 1.00 (Standard laptop / workstation GPUs like RTX 4060 / 5060)
      large  : base_c = 64, depth_mul = 1.33 (High-end workstation & server GPUs)
      xlarge : base_c = 80, depth_mul = 1.67 (Maximum capacity multi-task foundation)
    """
    configs = {
        "pico":   (8,  0.25),
        "femto":  (12, 0.33),
        "nano":   (16, 0.33),
        "micro":  (24, 0.50),
        "small":  (32, 0.67),
        "medium": (48, 1.00),
        "large":  (64, 1.33),
        "xlarge": (80, 1.67)
    }
    # For ultra-constrained models (pico and femto), disable heavy video intelligence heads by default for maximal speed
    if size.lower() in ("pico", "femto"):
        enable_pose = False
        enable_st_intel = False
        enable_prompt = False

    base, d_mul = configs.get(size.lower(), (48, 1.00))
    return Neuravex(
        num_classes=num_classes,
        base_c=base,
        depth_mul=d_mul,
        reg_max=reg_max,
        enable_pose=enable_pose,
        enable_st_intel=enable_st_intel,
        enable_prompt=enable_prompt
    )
