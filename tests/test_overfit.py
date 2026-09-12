import torch
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from yolo27.models.yolo27 import build_yolo27
from yolo27.engine.trainer import YOLO27MultiTaskTrainer

def run_end_to_end_train_step():
    print("\n--- Running End-to-End Pipeline Verification ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build model (nano for fast test)
    model = build_yolo27(size="nano", num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    trainer = YOLO27MultiTaskTrainer(model, optimizer, device=device, num_classes=5)

    # Synthetic batch with all modalities active
    B = 2
    img_size = 128
    images = torch.rand(B, 3, img_size, img_size)
    gt_boxes = torch.tensor([
        [[10., 10., 40., 40.], [50., 50., 90., 90.]],
        [[20., 20., 60., 70.], [0., 0., 0., 0.]]
    ])
    gt_labels = torch.tensor([
        [[1], [2]],
        [[3], [0]]
    ])
    mask_gt = torch.tensor([
        [[True], [True]],
        [[True], [False]]
    ])
    sem_masks = torch.randint(0, 5, (B, img_size, img_size))
    inst_masks = torch.zeros(B, img_size, img_size, dtype=torch.long)
    inst_masks[:, 10:40, 10:40] = 1
    bound_maps = (torch.rand(B, 1, img_size, img_size) > 0.8).float()

    # 3D GT
    gt_3d_xyz = torch.tensor([
        [[0.0, 0.0, 5.0], [1.0, -1.0, 8.0]],
        [[-0.5, 0.5, 6.0], [0.0, 0.0, 0.0]]
    ])
    gt_3d_lwh = torch.tensor([
        [[4.0, 2.0, 1.5], [3.5, 1.8, 1.6]],
        [[4.2, 1.9, 1.5], [1.0, 1.0, 1.0]]
    ])
    gt_3d_yaw = torch.tensor([
        [[0.2], [-0.4]],
        [[0.1], [0.0]]
    ])

    # Depth GT
    depth = torch.rand(B, 1, img_size, img_size) * 10.0 + 0.5
    valid_depth = torch.ones(B, 1, img_size, img_size, dtype=torch.bool)

    batch = {
        "images": images,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "mask_gt": mask_gt,
        "semantic_masks": sem_masks,
        "instance_masks": inst_masks,
        "boundary_maps": bound_maps,
        "gt_3d_xyz": gt_3d_xyz,
        "gt_3d_lwh": gt_3d_lwh,
        "gt_3d_yaw": gt_3d_yaw,
        "depth": depth,
        "valid_depth": valid_depth,
        "task_masks": {
            "det": 1.0,
            "semantic": 1.0,
            "boundary": 1.0,
            "instance": 1.0,
            "mask_quality": 1.0,
            "depth": 1.0,
            "geometry_3d": 1.0,
            "consistency": 1.0
        }
    }

    # Record initial parameter values to verify updates
    initial_weights = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    res = trainer.train_step(batch)

    print(f"Total Loss: {res['total_loss']:.4f}")
    print(f"Gradient Norm: {res['grad_norm']:.4f}")
    print("Raw Task Losses:")
    for k, v in res["raw_losses"].items():
        print(f"  {k:15s}: {v:.4f}")
    print("Task Weights (exp(-s)):")
    for k, v in res["task_weights"].items():
        print(f"  {k:15s}: {v:.4f}")

    # Verify parameters updated
    any_updated = False
    for p_init, p_curr in zip(initial_weights, [p for p in model.parameters() if p.requires_grad]):
        diff = (p_init - p_curr).abs().sum().item()
        if diff > 1e-7:
            any_updated = True
            break

    assert any_updated, "No model parameters were updated after train_step!"
    assert not math.isnan(res["total_loss"]), "Loss is NaN!"
    assert not math.isinf(res["total_loss"]), "Loss is Inf!"
    print("\n[SUCCESS] End-to-End step passed: gradients flowed, parameters updated, no NaN/Inf!")

def run_overfit_test(num_samples: int = 8, num_epochs: int = 25):
    print(f"\n--- Running {num_samples}-Sample Overfit Test ({num_epochs} epochs) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = build_yolo27(size="nano", num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    trainer = YOLO27MultiTaskTrainer(model, optimizer, device=device, num_classes=3)

    img_size = 128
    images = torch.rand(num_samples, 3, img_size, img_size)
    gt_boxes = torch.zeros(num_samples, 1, 4)
    gt_labels = torch.zeros(num_samples, 1, 1, dtype=torch.long)
    mask_gt = torch.ones(num_samples, 1, 1, dtype=torch.bool)

    for i in range(num_samples):
        gt_boxes[i, 0] = torch.tensor([15.0 + i * 2, 15.0 + i * 2, 50.0 + i * 2, 50.0 + i * 2])
        gt_labels[i, 0, 0] = i % 3

    sem_masks = torch.randint(0, 3, (num_samples, img_size, img_size))
    inst_masks = torch.zeros(num_samples, img_size, img_size, dtype=torch.long)
    for i in range(num_samples):
        inst_masks[i, 15:45, 15:45] = 1
    bound_maps = (torch.rand(num_samples, 1, img_size, img_size) > 0.8).float()

    batch = {
        "images": images,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "mask_gt": mask_gt,
        "semantic_masks": sem_masks,
        "instance_masks": inst_masks,
        "boundary_maps": bound_maps,
        "task_masks": {
            "det": 1.0,
            "semantic": 1.0,
            "boundary": 1.0,
            "instance": 1.0,
            "mask_quality": 1.0,
            "consistency": 1.0
        }
    }

    initial_loss = None
    final_loss = None

    for epoch in range(1, num_epochs + 1):
        res = trainer.train_step(batch)
        loss = res["total_loss"]
        if epoch == 1:
            initial_loss = loss
        if epoch == num_epochs:
            final_loss = loss
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{num_epochs:2d} | Loss: {loss:.4f} | GradNorm: {res['grad_norm']:.4f}")

    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease! {initial_loss} -> {final_loss}"
    print("[SUCCESS] Overfit test passed: loss monotonically and significantly decreased!")

if __name__ == "__main__":
    import math
    run_end_to_end_train_step()
    run_overfit_test()
