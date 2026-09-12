import torch
import math
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo27.geometry.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou_2d, bbox_ciou
from yolo27.geometry.oriented_iou3d import oriented_iou_3d, boxes3d_to_corners
from yolo27.geometry.camera import CameraIntrinsics, depth_to_inverse, inverse_to_depth
from yolo27.loss.assigner import TaskAlignedAssigner
from yolo27.loss.segmentation_loss import (
    multiclass_dice_loss, semantic_segmentation_loss,
    boundary_loss, mask_quality_loss, discriminative_instance_loss
)
from yolo27.loss.depth_3d_loss import scale_invariant_log_depth_loss, loss_3d_detection
from yolo27.data.augmentation import GeometricMultiViewAugment, invert_box_transform, invert_yaw_transform
from yolo27.models.yolo27 import build_yolo27

def test_box_ops():
    print("Testing 2D box operations & CIoU...")
    boxes = torch.tensor([[10., 10., 50., 50.], [20., 20., 80., 80.]])
    cxcywh = box_xyxy_to_cxcywh(boxes)
    xyxy_rec = box_cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(boxes, xyxy_rec, atol=1e-5), "Box conversion roundtrip failed!"

    iou = box_iou_2d(boxes[:1], boxes[:1])
    assert abs(iou.item() - 1.0) < 1e-4, f"Self IoU should be 1.0, got {iou.item()}"

    ciou = bbox_ciou(boxes[:1], boxes[:1])
    assert abs(ciou.item() - 1.0) < 1e-4, f"Self CIoU should be 1.0, got {ciou.item()}"
    print("  [PASS] 2D box operations & CIoU")

def test_oriented_3d_iou():
    print("Testing Oriented 3D IoU with yaw rotation...")
    # Two identical boxes: IoU should be 1.0
    c1 = torch.tensor([[0., 0., 5.]])
    lwh1 = torch.tensor([[4., 2., 1.5]])
    yaw1 = torch.tensor([0.])

    iou_self = oriented_iou_3d(c1, lwh1, yaw1, c1, lwh1, yaw1)
    assert abs(iou_self.item() - 1.0) < 1e-4, f"Self 3D IoU should be 1.0, got {iou_self.item()}"

    # Rotated box by 90 degrees around vertical axis: width and length swap
    yaw_90 = torch.tensor([math.pi / 2.0])
    iou_rot90 = oriented_iou_3d(c1, lwh1, yaw1, c1, lwh1, yaw_90)
    # Area of overlap is 2x2 = 4, individual volume is 4x2x1.5=12, union = 12+12-4*1.5 = 18 -> 6/18 = 0.333
    print(f"  Rotated 90 deg 3D IoU: {iou_rot90.item():.4f}")
    assert 0.0 < iou_rot90.item() < 1.0, "Rotated 3D IoU should be between 0 and 1"

    # Completely separated boxes: IoU should be 0.0
    c_far = torch.tensor([[100., 100., 100.]])
    iou_zero = oriented_iou_3d(c1, lwh1, yaw1, c_far, lwh1, yaw1)
    assert iou_zero.item() == 0.0, f"Disjoint boxes should have 0 IoU, got {iou_zero.item()}"
    print("  [PASS] Oriented 3D IoU")

def test_camera_and_depth():
    print("Testing Camera Intrinsics and Depth conversions...")
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=160.0)
    # Point at center (160, 160) with Z=5 should unproject to (0, 0, 5)
    u = torch.tensor([160.])
    v = torch.tensor([160.])
    z = torch.tensor([5.])
    xyz = K.unproject_points(u, v, z)
    assert torch.allclose(xyz, torch.tensor([[0., 0., 5.]]), atol=1e-4), f"Unprojected point mismatch: {xyz}"

    # Round-trip projection
    uv_rec = K.project_points(xyz)
    assert torch.allclose(uv_rec, torch.tensor([[160., 160.]]), atol=1e-4), f"Projected UV mismatch: {uv_rec}"

    # Depth & inverse depth
    rho = depth_to_inverse(z)
    z_rec = inverse_to_depth(rho)
    assert torch.allclose(z, z_rec, atol=1e-4), "Depth inverse roundtrip failed!"
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
    print(f"  Matched foreground anchors: {fg_mask.sum().item()}")

    # Multiclass Dice & CE loss
    logits = torch.randn(2, 5, 64, 64)
    target = torch.randint(0, 5, (2, 64, 64))
    l_sem = semantic_segmentation_loss(logits, target)
    assert not torch.isnan(l_sem) and l_sem.item() > 0, "Semantic loss failed!"

    # Boundary loss on raw logits
    b_logits = torch.randn(2, 1, 64, 64)
    b_target = (torch.rand(2, 1, 64, 64) > 0.8).float()
    l_bound = boundary_loss(b_logits, b_target)
    assert not torch.isnan(l_bound) and l_bound.item() > 0, "Boundary loss failed!"

    # Instance embedding loss
    inst_emb = torch.nn.functional.normalize(torch.randn(2, 16, 64, 64), dim=1)
    inst_ids = torch.zeros(2, 64, 64, dtype=torch.long)
    inst_ids[:, 10:25, 10:25] = 1
    inst_ids[:, 35:50, 35:50] = 2
    l_inst = discriminative_instance_loss(inst_emb, inst_ids)
    assert not torch.isnan(l_inst) and l_inst.item() >= 0, "Instance loss failed!"

    # Mask quality loss
    q_logits = torch.randn(2, 1, 64, 64)
    p_mask = torch.rand(2, 1, 64, 64)
    g_mask = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l_q = mask_quality_loss(q_logits, p_mask, g_mask)
    assert not torch.isnan(l_q) and l_q.item() >= 0, "Mask quality loss failed!"

    print("  [PASS] Assigner and segmentation losses")

def test_augmentation_transform_inversion():
    print("Testing Augmentation inverse transforms...")
    augmenter = GeometricMultiViewAugment(p_flip=1.0)
    x = torch.rand(2, 3, 64, 64)
    aug_x, inv_mat, is_hflip = augmenter(x)
    assert is_hflip.all(), "Should be flipped with p_flip=1.0"

    # Invert box
    boxes = torch.tensor([[[10., 10., 30., 40.]]])
    inv_boxes = invert_box_transform(boxes, is_hflip[:1], img_w=64.0)
    # Original width: 30-10 = 20. Inverted: x1_new = 64-30 = 34, x2_new = 64-10 = 54. Width = 20.
    assert inv_boxes[0, 0, 0].item() == 34.0 and inv_boxes[0, 0, 2].item() == 54.0, "Box flip inversion error!"

    # Invert yaw
    yaw = torch.tensor([[0.5]])
    inv_yaw = invert_yaw_transform(yaw, is_hflip[:1])
    assert abs(inv_yaw[0, 0].item() - (math.pi - 0.5)) < 1e-5, "Yaw flip inversion error!"
    print("  [PASS] Augmentation and transform inversion")

def test_model_forward():
    print("Testing YOLO27 v0.5 Model Forward Pass...")
    model = build_yolo27(size="small", num_classes=80)
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    
    expected_keys = [
        "class_logits", "pred_boxes", "pred_xyz", "pred_lwh", "pred_yaw_sincos",
        "semantic_masks", "boundary_map", "instance_embeddings", "mask_quality",
        "depth_inverse", "depth_map"
    ]
    for k in expected_keys:
        assert k in out, f"Missing output key: {k}"
    print(f"  Model output keys verified: {len(out.keys())} keys present.")
    print(f"  Anchor points total: {out['anchor_points'].shape[0]}")
    print("  [PASS] Model Forward")

if __name__ == "__main__":
    test_box_ops()
    test_oriented_3d_iou()
    test_camera_and_depth()
    test_assigner_and_losses()
    test_augmentation_transform_inversion()
    test_model_forward()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
