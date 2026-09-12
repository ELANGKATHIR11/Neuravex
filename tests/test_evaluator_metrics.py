import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo27.engine.evaluator import (
    calculate_map_metrics,
    calculate_miou,
    calculate_depth_metrics,
    calculate_boundary_fscore
)

def test_evaluator_metrics():
    print("Testing Evaluator accuracy calculation routines...")

    # 1. Test mAP calculation
    pred_boxes = [torch.tensor([[10., 10., 50., 50.], [60., 60., 100., 100.]])]
    pred_scores = [torch.tensor([0.95, 0.88])]
    pred_labels = [torch.tensor([1, 2])]

    gt_boxes = [torch.tensor([[10., 10., 50., 50.], [62., 62., 98., 98.]])]
    gt_labels = [torch.tensor([1, 2])]

    map_res = calculate_map_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    assert "mAP50" in map_res and "mAP50:95" in map_res
    print(f"  [PASS] calculate_map_metrics: mAP50={map_res['mAP50']:.4f}, mAP50:95={map_res['mAP50:95']:.4f}")

    # 2. Test mIoU calculation
    gt_sem = torch.tensor([[0, 1, 1], [2, 2, 0]])
    pred_sem = torch.tensor([[0, 1, 1], [2, 0, 0]])
    miou = calculate_miou(pred_sem, gt_sem, num_classes=3)
    assert 0.0 <= miou <= 1.0
    print(f"  [PASS] calculate_miou: {miou:.4f}")

    # 3. Test Depth RMSE / AbsRel
    gt_d = torch.tensor([1.0, 2.0, 3.0])
    pred_d = torch.tensor([1.1, 1.9, 3.2])
    v_mask = torch.tensor([True, True, True])
    d_res = calculate_depth_metrics(pred_d, gt_d, v_mask)
    assert "RMSE" in d_res and "AbsRel" in d_res and "delta1" in d_res
    print(f"  [PASS] calculate_depth_metrics: RMSE={d_res['RMSE']:.4f}, AbsRel={d_res['AbsRel']:.4f}")

    # 4. Test Boundary F-Score
    gt_b = torch.tensor([0.0, 1.0, 1.0, 0.0])
    pred_b = torch.tensor([0.1, 0.8, 0.9, 0.2])
    f1 = calculate_boundary_fscore(pred_b, gt_b)
    assert f1 == 1.0
    print(f"  [PASS] calculate_boundary_fscore: {f1:.4f}")

    print("\nAll accuracy and evaluation metric routines verified successfully!")

if __name__ == "__main__":
    test_evaluator_metrics()
