import torch
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo27.geometry.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou_2d, bbox_ciou
from yolo27.geometry.oriented_iou3d import oriented_iou_3d, boxes3d_to_corners
from yolo27.geometry.camera import CameraIntrinsics, depth_to_inverse, inverse_to_depth
from yolo27.loss.assigner import TaskAlignedAssigner
from yolo27.loss.detection_loss import DetectionLoss, dfl_loss
from yolo27.loss.segmentation_loss import (
    multiclass_dice_loss, semantic_segmentation_loss,
    boundary_loss, mask_quality_loss, discriminative_instance_loss
)
from yolo27.loss.depth_3d_loss import comprehensive_metric_depth_loss, loss_3d_detection
from yolo27.data.augmentation import (
    GeometricMultiViewAugment, invert_box_transform,
    invert_yaw_transform, invert_3d_center_transform
)
from yolo27.models.yolo27 import build_yolo27
from yolo27.engine.evaluator import (
    YOLO27InferencePostProcessor,
    calculate_map_metrics,
    calculate_miou,
    calculate_depth_metrics,
    calculate_boundary_fscore
)

def test_box_and_dfl():
    print("Testing 2D box operations & DFL...")
    boxes = torch.tensor([[10., 10., 50., 50.], [20., 20., 80., 80.]])
    cxcywh = box_xyxy_to_cxcywh(boxes)
    xyxy_rec = box_cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(boxes, xyxy_rec, atol=1e-5), "Box conversion roundtrip failed!"

    iou = box_iou_2d(boxes[:1], boxes[:1])
    assert abs(iou.item() - 1.0) < 1e-4, f"Self IoU should be 1.0, got {iou.item()}"

    ciou = bbox_ciou(boxes[:1], boxes[:1])
    assert abs(ciou.item() - 1.0) < 1e-4, f"Self CIoU should be 1.0, got {ciou.item()}"

    # DFL loss test
    pred_dist = torch.randn(2, 16)
    target_dist = torch.tensor([3.4, 7.8])
    l_dfl = dfl_loss(pred_dist, target_dist)
    assert not torch.isnan(l_dfl) and l_dfl.item() > 0, "DFL loss failed!"
    print("  [PASS] 2D box operations & DFL")

def test_oriented_3d_iou():
    print("Testing Oriented 3D IoU symmetry and bounds...")
    c1 = torch.tensor([[0., 0., 5.]])
    lwh1 = torch.tensor([[4., 2., 1.5]])
    yaw1 = torch.tensor([0.])

    # Self IoU
    iou_self = oriented_iou_3d(c1, lwh1, yaw1, c1, lwh1, yaw1)
    assert abs(iou_self.item() - 1.0) < 1e-4, f"Self 3D IoU should be 1.0, got {iou_self.item()}"

    # Symmetry: IoU(A, B) == IoU(B, A)
    c2 = torch.tensor([[0.5, -0.2, 5.1]])
    lwh2 = torch.tensor([[3.8, 2.1, 1.6]])
    yaw2 = torch.tensor([0.4])
    iou_ab = oriented_iou_3d(c1, lwh1, yaw1, c2, lwh2, yaw2)
    iou_ba = oriented_iou_3d(c2, lwh2, yaw2, c1, lwh1, yaw1)
    assert abs(iou_ab.item() - iou_ba.item()) < 1e-4, f"Symmetry failed: {iou_ab.item()} vs {iou_ba.item()}"

    # Rotated box by 90 degrees
    yaw_90 = torch.tensor([math.pi / 2.0])
    iou_rot90 = oriented_iou_3d(c1, lwh1, yaw1, c1, lwh1, yaw_90)
    assert 0.0 < iou_rot90.item() < 1.0, "Rotated 3D IoU should be between 0 and 1"

    # Disjoint boxes
    c_far = torch.tensor([[100., 100., 100.]])
    iou_zero = oriented_iou_3d(c1, lwh1, yaw1, c_far, lwh1, yaw1)
    assert iou_zero.item() == 0.0, f"Disjoint boxes should have 0 IoU, got {iou_zero.item()}"
    print("  [PASS] Oriented 3D IoU symmetry and bounds")

def test_camera_and_depth():
    print("Testing Camera Intrinsics and Depth conversions...")
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=160.0)
    u = torch.tensor([160.])
    v = torch.tensor([160.])
    z = torch.tensor([5.])
    xyz = K.unproject_points(u, v, z)
    assert torch.allclose(xyz, torch.tensor([[0., 0., 5.]]), atol=1e-4)

    uv_rec = K.project_points(xyz)
    assert torch.allclose(uv_rec, torch.tensor([[160., 160.]]), atol=1e-4)

    rho = depth_to_inverse(z)
    z_rec = inverse_to_depth(rho)
    assert torch.allclose(z, z_rec, atol=1e-4)
    print("  [PASS] Camera and Depth")

def test_assigner_and_losses():
    print("Testing TaskAlignedAssigner and multi-task losses...")
    B, N, C = 2, 100, 80
    scores = torch.sigmoid(torch.randn(B, N, C))
    boxes = torch.rand(B, N, 4) * 320
    boxes[..., 2:] = boxes[..., :2] + torch.rand(B, N, 2) * 50
    anc = torch.rand(N, 2) * 320

    gt_boxes = torch.tensor([[[20., 20., 60., 60.], [0., 0., 0., 0.]],
                            [[50., 50., 100., 100.], [0., 0., 0., 0.]]])
    gt_labels = torch.tensor([[[1], [0]], [[5], [0]]])
    mask_gt = torch.tensor([[[True], [False]], [[True], [False]]])

    assigner = TaskAlignedAssigner(topk=5, num_classes=C)
    t_labels, t_bboxes, t_scores, fg_mask, t_gt_idx = assigner(scores, boxes, anc, gt_labels, gt_boxes, mask_gt)

    assert fg_mask.any(), "Assigner should match at least one foreground anchor!"

    # Multi-class Dice & CE
    logits = torch.randn(2, 5, 64, 64)
    target = torch.randint(0, 5, (2, 64, 64))
    l_sem = semantic_segmentation_loss(logits, target)
    assert not torch.isnan(l_sem) and l_sem.item() > 0

    # Boundary loss
    b_logits = torch.randn(2, 1, 64, 64)
    b_target = (torch.rand(2, 1, 64, 64) > 0.8).float()
    l_bound = boundary_loss(b_logits, b_target)
    assert not torch.isnan(l_bound) and l_bound.item() > 0

    # Instance loss
    inst_emb = torch.nn.functional.normalize(torch.randn(2, 16, 64, 64), dim=1)
    inst_ids = torch.zeros(2, 64, 64, dtype=torch.long)
    inst_ids[:, 10:25, 10:25] = 1
    inst_ids[:, 35:50, 35:50] = 2
    l_inst = discriminative_instance_loss(inst_emb, inst_ids)
    assert not torch.isnan(l_inst) and l_inst.item() >= 0

    # Comprehensive metric depth loss (SILog + L1 + Gradient)
    pred_depth = torch.rand(2, 1, 64, 64) * 10.0 + 0.1
    gt_depth = torch.rand(2, 1, 64, 64) * 10.0 + 0.1
    v_depth = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    l_depth = comprehensive_metric_depth_loss(pred_depth, gt_depth, v_depth)
    assert not torch.isnan(l_depth) and l_depth.item() > 0

    print("  [PASS] Assigner and multi-task losses")

def test_model_forward():
    print("Testing YOLO27 v0.6 Model Forward with Depth Scaling...")
    # Test depth_mul scaling across sizes
    model_nano = build_yolo27(size="nano", num_classes=80)
    model_small = build_yolo27(size="small", num_classes=80)
    model_med = build_yolo27(size="medium", num_classes=80)

    p_nano = sum(p.numel() for p in model_nano.parameters())
    p_small = sum(p.numel() for p in model_small.parameters())
    p_med = sum(p.numel() for p in model_med.parameters())
    assert p_nano < p_small < p_med, "Scaling hierarchy failed!"

    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        out = model_med(x)

    expected_keys = [
        "class_logits", "pred_box_dist", "pred_boxes", "pred_xyz", "pred_lwh", "pred_yaw_sincos",
        "semantic_masks", "boundary_map", "instance_embeddings", "mask_quality",
        "proto_masks", "depth_inverse", "depth_map"
    ]
    for k in expected_keys:
        assert k in out, f"Missing output key: {k}"
    print(f"  Verified {len(expected_keys)} multi-task output keys.")
    print("  [PASS] Model Forward")

if __name__ == "__main__":
    test_box_and_dfl()
    test_oriented_3d_iou()
    test_camera_and_depth()
    test_assigner_and_losses()
    test_model_forward()
    print("\nALL COMPONENT TESTS PASSED SUCCESSFULLY!")
