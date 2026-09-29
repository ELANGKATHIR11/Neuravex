"""
Neuravex Native Precision Perception Pipeline.

RULES ENFORCED:
 - NO external detector (YOLO, etc.) in native mode.
 - NO fabricated depth, dimensions, coordinates, confidence, or labels.
 - Instance masks assembled from proto_masks × mask_coefficients (1:1 per detection).
 - Metric depth is the DEM head output directly; marked UNVALIDATED without calibration data.
 - LWH and yaw come from trained Neuravex heads; never heuristic multipliers.
 - 3D cuboid corners computed mathematically: XYZ + LWH + rotation → 8 corners → project with K.
 - Missing data → None / explicit blocker flag. Never invented.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union

from ..models.neuravex import build_neuravex
from ..geometry.camera import CameraIntrinsics
from ..geometry.oriented_iou3d import boxes3d_to_corners
from ..core import ObjectState, FrameState


# ---------------------------------------------------------------------------
# 3D Box Projection Helpers
# ---------------------------------------------------------------------------

def project_3d_corners_to_image(
    center: np.ndarray,    # (3,) XYZ in camera frame
    lwh: np.ndarray,       # (3,) L, W, H in meters
    yaw: float,            # radians
    intrinsics: CameraIntrinsics,
) -> Optional[np.ndarray]:
    """
    Mathematical 3D bounding box projection:
      1. Generate 8 corners from center + dimensions + rotation.
      2. Rotate corners by yaw around Y-axis.
      3. Translate to world position.
      4. Project to image plane using pinhole K.
    Returns (8, 2) integer pixel coordinates, or None if behind camera.
    """
    if center[2] <= 0.05:
        return None

    # Use the model's boxes3d_to_corners (expects tensors)
    c_t = torch.tensor(center, dtype=torch.float32).unsqueeze(0)
    lwh_t = torch.tensor(lwh, dtype=torch.float32).unsqueeze(0)
    yaw_t = torch.tensor([yaw], dtype=torch.float32)

    corners_3d = boxes3d_to_corners(c_t, lwh_t, yaw_t)  # (1, 8, 3)
    corners_3d = corners_3d.squeeze(0)  # (8, 3)

    # Filter: all corners must be in front of camera
    if (corners_3d[:, 2] < 0.01).any():
        return None

    # Project using pinhole: u = X*fx/Z + cx, v = Y*fy/Z + cy
    corners_2d = intrinsics.project_points(corners_3d)  # (8, 2)
    return corners_2d.numpy().astype(np.int32)


def render_3d_wireframe(
    img: np.ndarray,
    corners_2d: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.12,
):
    """Renders a shaded 3D wireframe perspective cube from projected corners."""
    overlay = img.copy()

    # Shaded top and right faces
    top_poly = np.array([corners_2d[0], corners_2d[1], corners_2d[5], corners_2d[4]])
    right_poly = np.array([corners_2d[1], corners_2d[2], corners_2d[6], corners_2d[5]])
    cv2.fillPoly(overlay, [top_poly], color)
    cv2.fillPoly(overlay, [right_poly], color)
    cv2.addWeighted(overlay, fill_alpha, img, 1.0 - fill_alpha, 0, img)

    # Back face
    cv2.polylines(img, [corners_2d[4:8]], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    # Connecting pillars
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i + 4]), color, thickness, cv2.LINE_AA)
    # Front face
    cv2.polylines(img, [corners_2d[0:4]], isClosed=True, color=color, thickness=thickness + 1, lineType=cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Instance Mask Assembly
# ---------------------------------------------------------------------------

def assemble_instance_masks(
    proto_masks: torch.Tensor,   # (B, num_proto, H, W)
    mask_coeffs: torch.Tensor,   # (N_det, num_proto)
    det_boxes: torch.Tensor,     # (N_det, 4) xyxy
    img_h: int,
    img_w: int,
) -> List[np.ndarray]:
    """
    Assemble per-detection instance masks from prototype masks and learned coefficients.
    mask_i = sigmoid(coeffs_i @ proto_masks)  [YOLACT-style]

    Returns list of (img_h, img_w) binary uint8 masks, one per detection.
    Each mask is cropped to its detection bounding box to avoid cross-object leakage.
    """
    masks_out = []
    if proto_masks is None or mask_coeffs is None or len(det_boxes) == 0:
        return masks_out

    protos = proto_masks[0]  # (num_proto, Hp, Wp) — first batch element
    Hp, Wp = protos.shape[1], protos.shape[2]

    # Compute all masks at prototype resolution
    # mask_logits = coeffs @ protos  → (N_det, Hp, Wp)
    mask_logits = torch.einsum("nc,chw->nhw", mask_coeffs, protos)
    mask_probs = torch.sigmoid(mask_logits)

    for i in range(len(det_boxes)):
        # Scale box coordinates to prototype resolution
        box = det_boxes[i].detach().cpu().numpy()
        sx = Wp / img_w
        sy = Hp / img_h
        px1 = max(0, int(box[0] * sx))
        py1 = max(0, int(box[1] * sy))
        px2 = min(Wp, int(box[2] * sx))
        py2 = min(Hp, int(box[3] * sy))

        # Create cropped mask at proto resolution
        mask_proto_res = torch.zeros((Hp, Wp), device=protos.device)
        if px2 > px1 and py2 > py1:
            mask_proto_res[py1:py2, px1:px2] = mask_probs[i, py1:py2, px1:px2]

        # Upsample to image resolution
        mask_full = F.interpolate(
            mask_proto_res.unsqueeze(0).unsqueeze(0),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        binary = (mask_full > 0.5).cpu().numpy().astype(np.uint8)
        masks_out.append(binary)

    return masks_out


# ---------------------------------------------------------------------------
# Precision Perception Pipeline
# ---------------------------------------------------------------------------

class PrecisionPerceptionPipeline:
    """
    Native Neuravex perception pipeline.
    RGB → Neuravex → Detect → InstanceMask → MetricDepth → XYZ → LWH+Rotation → 3DBox.

    No external detector. No fabricated data.
    """

    PALETTE: List[Tuple[int, int, int]] = [
        (0, 235, 255),    # Cyan
        (255, 105, 180),  # Magenta
        (50, 255, 120),   # Neon Green
        (255, 180, 0),    # Amber
        (130, 100, 255),  # Lavender
        (0, 255, 200),    # Aquamarine
        (255, 120, 50),   # Coral
        (255, 220, 80),   # Lemon
        (180, 50, 255),   # Violet
        (255, 80, 80),    # Rose
    ]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        model_size: str = "nano",
        num_classes: int = 80,
        class_names: Optional[List[str]] = None,
    ):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.conf_thresh = conf_thresh
        self.num_classes = num_classes
        self.class_names = class_names

        # Build Neuravex model — NO external detector
        self.model = build_neuravex(size=model_size, num_classes=num_classes).to(self.device)

        # Load weights if available
        if weights_path and os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)

        self.model.eval()

    def _get_class_name(self, class_id: int) -> str:
        if self.class_names and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    @torch.no_grad()
    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        output_vis_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
        input_size: int = 640,
    ) -> Dict[str, Any]:
        """
        Native Neuravex precision perception.

        Args:
            image_input: Path to image file or BGR numpy array.
            output_vis_path: Optional path to save visualization.
            output_json_path: Optional path to save JSON results.
            intrinsics: Camera intrinsics. If None, estimated from image (marked UNVALIDATED).
            input_size: Model input resolution (default 640).

        Returns:
            Dictionary with FrameState-compatible structured results.
        """
        # ---- 1. Load Image ----
        if isinstance(image_input, str):
            orig_bgr = cv2.imread(image_input)
            if orig_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image_input}")
            img_filename = os.path.basename(image_input)
        else:
            orig_bgr = image_input.copy()
            img_filename = "memory_array.jpg"

        orig_h, orig_w = orig_bgr.shape[:2]
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # ---- 2. Intrinsics ----
        intrinsics_provided = intrinsics is not None
        if intrinsics is None:
            # Estimate from image — explicitly UNVALIDATED
            intrinsics = CameraIntrinsics(
                fx=orig_w * 1.15, fy=orig_w * 1.15,
                cx=orig_w / 2.0, cy=orig_h / 2.0,
                device="cpu",
            )

        # Intrinsics scaled to model input
        sx = input_size / orig_w
        sy = input_size / orig_h
        K_model = intrinsics.scale(sx, sy)
        K_model_dev = CameraIntrinsics(
            fx=K_model.fx, fy=K_model.fy, cx=K_model.cx, cy=K_model.cy,
            device=str(self.device),
        )

        # ---- 3. Preprocess ----
        img_resized = cv2.resize(orig_rgb, (input_size, input_size))
        t_img = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )

        # ---- 4. Forward Pass — ALL tasks via Neuravex ----
        model_out = self.model(t_img, intrinsics=K_model_dev)

        # ---- 5. Post-Process Detections (NMS) ----
        pred_cls = torch.sigmoid(model_out["class_logits"])  # (1, N, C)
        pred_boxes = model_out["pred_boxes"]                  # (1, N, 4)
        pred_mask_coeffs = model_out.get("pred_mask_coeffs")  # (1, N, num_proto)
        pred_xyz = model_out.get("pred_xyz")
        pred_lwh = model_out.get("pred_lwh")
        pred_yaw_sc = model_out.get("pred_yaw_sincos")
        depth_map = model_out.get("depth_map")
        depth_conf_map = model_out.get("depth_confidence")
        terrain_elev = model_out.get("terrain_elevation")
        proto_masks = model_out.get("proto_masks")

        # Recover yaw from sin/cos
        if pred_yaw_sc is not None:
            yaw_norm = F.normalize(pred_yaw_sc, dim=-1)
            recovered_yaw = torch.atan2(yaw_norm[..., 0], yaw_norm[..., 1]).unsqueeze(-1)
        else:
            recovered_yaw = None

        # Score-based filtering + NMS
        scores, labels = pred_cls[0].max(dim=-1)
        keep_mask = scores > self.conf_thresh
        if not keep_mask.any():
            return self._build_empty_result(img_filename, orig_w, orig_h, intrinsics_provided)

        f_boxes = pred_boxes[0, keep_mask]
        f_scores = scores[keep_mask]
        f_labels = labels[keep_mask]
        f_coeffs = pred_mask_coeffs[0, keep_mask] if pred_mask_coeffs is not None else None
        f_xyz = pred_xyz[0, keep_mask] if pred_xyz is not None else None
        f_lwh = pred_lwh[0, keep_mask] if pred_lwh is not None else None
        f_yaw = recovered_yaw[0, keep_mask] if recovered_yaw is not None else None

        # Class-aware batched NMS
        from torchvision.ops import batched_nms
        keep = batched_nms(f_boxes, f_scores, f_labels, iou_threshold=0.45)
        keep = keep[:300]

        det_boxes = f_boxes[keep]
        det_scores = f_scores[keep]
        det_labels = f_labels[keep]
        det_coeffs = f_coeffs[keep] if f_coeffs is not None else None
        det_xyz = f_xyz[keep] if f_xyz is not None else None
        det_lwh = f_lwh[keep] if f_lwh is not None else None
        det_yaw = f_yaw[keep] if f_yaw is not None else None

        # ---- 6. Instance Mask Assembly ----
        instance_masks = assemble_instance_masks(
            proto_masks, det_coeffs, det_boxes, input_size, input_size,
        )
        masks_available = len(instance_masks) == len(det_boxes)
        mask_source = "proto_assembly" if masks_available else "none"

        # ---- 7. Depth Map Processing ----
        from .tracker import robust_mask_depth_estimator

        depth_map_np = None
        depth_conf_np = None
        if depth_map is not None:
            depth_map_np = depth_map[0, 0]  # (Hm, Wm) tensor on device
            if depth_conf_map is not None:
                depth_conf_np = depth_conf_map[0, 0]

        # ---- 8. Build ObjectState for Each Detection ----
        objects: List[ObjectState] = []
        vis_canvas = orig_bgr.copy() if output_vis_path else None

        for i in range(len(det_boxes)):
            box = det_boxes[i].cpu().numpy()
            score = float(det_scores[i].item())
            label_id = int(det_labels[i].item())

            # Scale box coordinates to original image
            ox1 = float(box[0] * orig_w / input_size)
            oy1 = float(box[1] * orig_h / input_size)
            ox2 = float(box[2] * orig_w / input_size)
            oy2 = float(box[3] * orig_h / input_size)
            bbox_orig = [ox1, oy1, ox2, oy2]

            # ---- Instance Mask ----
            mask_orig = None
            mask_conf = 0.0
            if masks_available and i < len(instance_masks):
                mask_model = instance_masks[i]
                mask_orig = cv2.resize(mask_model, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask_conf = float(score)  # mask quality tied to detection confidence

            # ---- Depth from DEM Head ----
            z_est = None
            u_center = None
            v_center = None
            d_conf = 0.0
            depth_src = "UNVALIDATED"

            if depth_map_np is not None:
                if mask_orig is not None and mask_orig.any():
                    # Resize mask to depth map resolution for robust estimation
                    Hd, Wd = depth_map_np.shape
                    mask_depth_res = torch.from_numpy(
                        cv2.resize(mask_orig, (Wd, Hd), interpolation=cv2.INTER_NEAREST)
                    ).to(depth_map_np.device).bool()
                else:
                    # Box-region fallback (not ideal, marked accordingly)
                    Hd, Wd = depth_map_np.shape
                    mask_depth_res = torch.zeros((Hd, Wd), dtype=torch.bool, device=depth_map_np.device)
                    bx1 = max(0, int(box[0] * Wd / input_size))
                    by1 = max(0, int(box[1] * Hd / input_size))
                    bx2 = min(Wd, int(box[2] * Wd / input_size))
                    by2 = min(Hd, int(box[3] * Hd / input_size))
                    if bx2 > bx1 and by2 > by1:
                        mask_depth_res[by1:by2, bx1:bx2] = True
                    mask_source_i = "box_fallback" if not masks_available else mask_source

                z_est, u_center, v_center, d_conf = robust_mask_depth_estimator(
                    depth_map_np, mask_depth_res,
                    depth_conf_np if depth_conf_np is not None else None,
                )
                if not math.isnan(z_est):
                    depth_src = "dem_head"
                else:
                    z_est = None

            # ---- XYZ from depth + intrinsics ----
            xyz_cam = None
            distance = None
            if z_est is not None:
                # Use depth centroid pixel coords for unprojection
                uc = u_center if u_center is not None and not math.isnan(u_center) else (box[0] + box[2]) / 2
                vc = v_center if v_center is not None and not math.isnan(v_center) else (box[1] + box[3]) / 2
                # Scale centroid back to original image coords for unprojection
                uc_orig = uc * orig_w / (depth_map_np.shape[1] if depth_map_np is not None else input_size)
                vc_orig = vc * orig_h / (depth_map_np.shape[0] if depth_map_np is not None else input_size)
                x_cam = float((uc_orig - intrinsics.cx) * z_est / intrinsics.fx)
                y_cam = float((vc_orig - intrinsics.cy) * z_est / intrinsics.fy)
                xyz_cam = [x_cam, y_cam, float(z_est)]
                distance = float(math.sqrt(x_cam**2 + y_cam**2 + z_est**2))
            elif det_xyz is not None:
                # Use detection head's raw XYZ prediction
                xyz_cam = det_xyz[i].cpu().numpy().tolist()
                distance = float(np.linalg.norm(xyz_cam))
                depth_src = "detection_head"

            # ---- LWH from model head (no heuristics) ----
            lwh_vals = None
            if det_lwh is not None:
                raw_lwh = det_lwh[i].cpu().numpy()
                # LWH comes from the model as exp-clamped positive values
                lwh_vals = [max(0.01, float(raw_lwh[0])),
                            max(0.01, float(raw_lwh[1])),
                            max(0.01, float(raw_lwh[2]))]

            # ---- Yaw from model head ----
            yaw_val = None
            if det_yaw is not None:
                yaw_val = float(det_yaw[i, 0].item())

            # ---- 3D Box Projection ----
            corners_3d = None
            corners_2d = None
            if xyz_cam is not None and lwh_vals is not None and yaw_val is not None:
                corners_2d = project_3d_corners_to_image(
                    np.array(xyz_cam), np.array(lwh_vals), yaw_val, intrinsics,
                )
                if corners_2d is not None:
                    c_t = torch.tensor(xyz_cam, dtype=torch.float32).unsqueeze(0)
                    l_t = torch.tensor(lwh_vals, dtype=torch.float32).unsqueeze(0)
                    y_t = torch.tensor([yaw_val], dtype=torch.float32)
                    corners_3d = boxes3d_to_corners(c_t, l_t, y_t).squeeze(0).numpy()

            # ---- Uncertainty from model ----
            xyz_unc = None
            lwh_unc = None
            log_sigma_xyz = model_out.get("log_sigma_xyz")
            log_sigma_lwh = model_out.get("log_sigma_lwh")
            if log_sigma_xyz is not None and keep_mask.any():
                # Get per-detection uncertainties
                unc_xyz_all = log_sigma_xyz[0, keep_mask][keep]
                if i < len(unc_xyz_all):
                    xyz_unc = torch.exp(0.5 * unc_xyz_all[i]).cpu().numpy().tolist()
            if log_sigma_lwh is not None and keep_mask.any():
                unc_lwh_all = log_sigma_lwh[0, keep_mask][keep]
                if i < len(unc_lwh_all):
                    lwh_unc = torch.exp(0.5 * unc_lwh_all[i]).cpu().numpy().tolist()

            # ---- Terrain / Height Above Ground ----
            terrain_val = None
            hag = None
            if terrain_elev is not None and xyz_cam is not None:
                te_map = terrain_elev[0, 0]
                Ht, Wt = te_map.shape
                tc = int((box[0] + box[2]) / 2 * Wt / input_size)
                tr = int((box[1] + box[3]) / 2 * Ht / input_size)
                tc = max(0, min(Wt - 1, tc))
                tr = max(0, min(Ht - 1, tr))
                terrain_val = float(te_map[tr, tc].item())
                if lwh_vals is not None:
                    hag = float(xyz_cam[1] - lwh_vals[2] / 2 - terrain_val) if terrain_val is not None else None

            # ---- Build ObjectState ----
            obj = ObjectState(
                id=i + 1,
                class_id=label_id,
                class_name=self._get_class_name(label_id),
                score=round(score, 4),
                bbox2d=[round(v, 1) for v in bbox_orig],
                mask=mask_orig,
                mask_confidence=round(mask_conf, 3),
                mask_source=mask_source if masks_available else "box_fallback",
                depth=round(z_est, 3) if z_est is not None else None,
                depth_confidence=round(d_conf, 3),
                depth_source=depth_src,
                xyz=[round(v, 3) for v in xyz_cam] if xyz_cam else None,
                lwh=[round(v, 3) for v in lwh_vals] if lwh_vals else None,
                yaw=round(yaw_val, 4) if yaw_val is not None else None,
                bbox3d_corners=corners_3d,
                bbox3d_projected=corners_2d,
                geometry_confidence=round(d_conf * score, 3),
                distance=round(distance, 3) if distance is not None else None,
                xyz_uncertainty=xyz_unc,
                lwh_uncertainty=lwh_unc,
                terrain_elevation=round(terrain_val, 3) if terrain_val is not None else None,
                height_above_ground=round(hag, 3) if hag is not None else None,
            )
            objects.append(obj)

            # ---- Visualization ----
            if vis_canvas is not None:
                color = self.PALETTE[i % len(self.PALETTE)]
                ix1, iy1, ix2, iy2 = int(ox1), int(oy1), int(ox2), int(oy2)

                # Instance mask overlay
                if mask_orig is not None:
                    tint = np.zeros_like(orig_bgr)
                    tint[mask_orig > 0] = list(color)
                    cv2.addWeighted(tint, 0.35, vis_canvas, 1.0, 0, vis_canvas)
                    contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_canvas, contours, -1, color, 2, cv2.LINE_AA)

                # 3D wireframe cube
                if corners_2d is not None:
                    render_3d_wireframe(vis_canvas, corners_2d, color, thickness=2, fill_alpha=0.12)
                else:
                    cv2.rectangle(vis_canvas, (ix1, iy1), (ix2, iy2), color, 2, cv2.LINE_AA)

                # Dynamic HUD badge
                self._render_hud_badge(vis_canvas, obj, color, orig_w, orig_h)

        # ---- 9. Build FrameState ----
        routing_stats = None
        rs = model_out.get("routing_stats")
        if rs:
            routing_stats = {
                "active_ratio": float(rs.get("active_ratio", 0)),
                "expected_compute": float(rs.get("expected_compute", 1.0)) if hasattr(rs.get("expected_compute", 1.0), 'item') else float(rs.get("expected_compute", 1.0)),
                "fully_executed": rs.get("fully_executed", True),
            }

        frame = FrameState(
            frame_id=0,
            image_width=orig_w,
            image_height=orig_h,
            objects=objects,
            total_visible=len(objects),
            depth_validated=False,  # Requires real calibration data to set True
            masks_validated=False,  # Requires trained mask supervision to set True
            intrinsics_provided=intrinsics_provided,
            routing_stats=routing_stats,
        )

        # ---- 10. Save Outputs ----
        if output_vis_path and vis_canvas is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_vis_path)), exist_ok=True)
            cv2.imwrite(output_vis_path, vis_canvas)

        result = frame.to_dict()
        result["image_filename"] = img_filename
        # Backward-compatible aliases
        result["total_animals_counted"] = len(objects)
        result["detected_objects"] = [o.to_dict() for o in objects]
        result["visualization_path"] = output_vis_path
        result["telemetry_path"] = output_json_path

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    def _build_empty_result(self, filename, w, h, intr_provided):
        frame = FrameState(
            image_width=w, image_height=h, total_visible=0,
            intrinsics_provided=intr_provided,
        )
        result = frame.to_dict()
        result["image_filename"] = filename
        result["total_animals_counted"] = 0
        result["detected_objects"] = []
        result["visualization_path"] = None
        result["telemetry_path"] = None
        return result

    @staticmethod
    def _render_hud_badge(
        canvas: np.ndarray,
        obj: ObjectState,
        color: Tuple[int, int, int],
        img_w: int,
        img_h: int,
    ):
        """Render a collision-free HUD badge for an object. Position is dynamic, not hardcoded."""
        if obj.bbox2d is None:
            return

        x1, y1, x2, y2 = [int(v) for v in obj.bbox2d]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.40
        thick = 1

        # Build text lines
        lines = [f"#{obj.id} {obj.class_name} | {obj.score:.2f}"]
        if obj.lwh:
            lines.append(f"LWH: {obj.lwh[0]:.2f} x {obj.lwh[1]:.2f} x {obj.lwh[2]:.2f} m")
        if obj.xyz:
            lines.append(f"XYZ: ({obj.xyz[0]:+.2f}, {obj.xyz[1]:+.2f}, {obj.xyz[2]:.2f}) m")
        if obj.distance is not None:
            lines.append(f"Dist: {obj.distance:.2f}m")
        if obj.depth_source == "UNVALIDATED":
            lines.append("[depth: UNVALIDATED]")

        sizes = [cv2.getTextSize(t, font, scale, thick)[0] for t in lines]
        bw = max(s[0] for s in sizes) + 16
        bh = sum(s[1] for s in sizes) + 8 * len(lines) + 8

        # Position badge above the box, fallback below if clipped
        bx = max(5, min(img_w - bw - 5, (x1 + x2) // 2 - bw // 2))
        by = y1 - bh - 10
        if by < 5:
            by = min(img_h - bh - 5, y2 + 10)

        # Leader line from badge center to box top center
        tx, ty = (x1 + x2) // 2, y1
        cv2.line(canvas, (bx + bw // 2, by + bh // 2), (tx, ty), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 3, color, -1, cv2.LINE_AA)

        # Badge background
        cv2.rectangle(canvas, (bx + 1, by + 1), (bx + bw + 1, by + bh + 1), (0, 0, 0), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (18, 20, 26), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 1)

        # Text
        y_off = by + 16
        for j, line in enumerate(lines):
            text_color = color if j == 0 else (220, 230, 255)
            cv2.putText(canvas, line, (bx + 8, y_off), font, scale, text_color, thick, cv2.LINE_AA)
            y_off += sizes[j][1] + 8
