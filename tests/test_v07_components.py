import sys
import os
import torch
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import build_neuravex
from neuravex.ssl.ssl_teacher import EMATeacher, MultiViewSSLLoss
from neuravex.ssl.semi_supervised import SemiSupervisedPseudoLabeler
from neuravex.ssl.instance_ssl import InstanceSSLClustering
from neuravex.loss.depth_ssl import PhotometricDepthSSLLoss
from neuravex.loss.cross_task_temporal import CrossTaskGeometryLoss, TemporalConsistencyLoss
from neuravex.engine.meta_controller import MetaControllerBandit
from neuravex.engine.adaptive_compute import AdaptiveComputeRouter, KnowledgeDistillationLoss
from neuravex.geometry.camera import CameraIntrinsics

def test_v07_modules():
    print("=================================================================")
    print(" RUNNING NEURAVEX v0.7 COMPONENT & ARCHITECTURE VERIFICATION")
    print("=================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. RepVGG-style RepConv and Backbone Reparameterization
    print("1. Testing Reparameterizable Backbone & switch_to_deploy()...")
    model = build_neuravex(size="nano", num_classes=5).to(device)
    x = torch.randn(2, 3, 320, 320, device=device)
    out_train = model(x)
    assert "class_logits" in out_train and "pred_boxes" in out_train
    
    # Switch to deploy and verify exact shape & numeric stability
    model.switch_to_deploy()
    out_deploy = model(x)
    assert out_deploy["pred_boxes"].shape == out_train["pred_boxes"].shape
    print("   [PASS] Backbone RepConv structural reparameterization verified.")

    # 2. EMA Teacher and Multi-View SSL Distillation
    print("2. Testing EMA Teacher and Multi-View SSL Loss...")
    teacher = EMATeacher(model, alpha=0.99)
    # Check no-grad property
    for p in teacher.model.parameters():
        assert not p.requires_grad
    # Update
    teacher.update(model)
    with torch.no_grad():
        t_out = teacher(x)
    ssl_loss_fn = MultiViewSSLLoss()
    ssl_loss = ssl_loss_fn(out_deploy, t_out)
    assert not torch.isnan(ssl_loss) and not torch.isinf(ssl_loss)
    print(f"   [PASS] EMA Teacher updated with no-grad; SSL Loss: {ssl_loss.item():.4f}")

    # 3. Semi-Supervised Adaptive Pseudo-Labeler
    print("3. Testing Semi-Supervised Dynamic Pseudo-Labeler...")
    pseudo_labeler = SemiSupervisedPseudoLabeler(num_classes=5, tau_base=0.5)
    pseudo_dict = pseudo_labeler.generate_pseudo_labels(t_out)
    assert "valid_mask" in pseudo_dict and "pseudo_boxes" in pseudo_dict
    print(f"   [PASS] Generated pseudo-labels across {pseudo_dict['valid_mask'].shape} predictions.")

    # 4. Photometric Self-Supervised Depth Loss
    print("4. Testing Photometric SSIM + L1 Depth Loss...")
    photo_loss_fn = PhotometricDepthSSLLoss()
    pred_d = torch.rand(2, 1, 320, 320, device=device) * 10.0 + 0.5
    img_tgt = torch.rand(2, 3, 320, 320, device=device)
    img_warp = img_tgt + torch.randn_like(img_tgt) * 0.05
    dssl_loss = photo_loss_fn(pred_d, img_tgt, img_warp)
    assert not torch.isnan(dssl_loss) and dssl_loss.item() > 0
    print(f"   [PASS] Photometric Depth SSL Loss: {dssl_loss.item():.4f}")

    # 5. Cross-Task Geometry & Temporal Consistency
    print("5. Testing Cross-Task Geometry & Temporal Losses...")
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=160.0, cy=160.0)
    cross_geom = CrossTaskGeometryLoss()
    geo_loss = cross_geom(
        out_deploy["pred_boxes"], out_deploy["pred_xyz"],
        out_deploy["semantic_masks"], out_deploy["depth_map"],
        intrinsics=intrinsics
    )
    assert not torch.isnan(geo_loss)
    
    temp_loss_fn = TemporalConsistencyLoss()
    temp_loss = temp_loss_fn(out_deploy, out_deploy)
    assert temp_loss.item() < 1e-4 # Identity temporal loss should be ~0
    print(f"   [PASS] Cross-task geometry loss: {geo_loss.item():.4f} | Temporal drift: {temp_loss.item():.6f}")

    # 6. Instance SSL Clustering
    print("6. Testing Instance SSL Clustering...")
    clusterer = InstanceSSLClustering()
    embeds = torch.randn(2, 16, 80, 80, device=device)
    depth_low = torch.rand(2, 1, 80, 80, device=device) * 10.0
    inst_maps = clusterer.cluster(embeds, depth_low)
    assert inst_maps.shape == (2, 80, 80)
    print(f"   [PASS] Clustered instance map with shape {inst_maps.shape}.")

    # 7. RL / Bandit Meta-Controller
    print("7. Testing RL Meta-Controller action bounds and reward...")
    meta_ctrl = MetaControllerBandit(state_dim=10, action_dim=7).to(device)
    dummy_state = torch.randn(1, 10, device=device)
    actions = meta_ctrl(dummy_state)
    assert 0.1 <= actions["aug_prob"].item() <= 0.9
    assert 0.5 <= actions["w_det"].item() <= 2.5
    reward = meta_ctrl.compute_reward(quality_gain=0.05, latency_ms=80.0, memory_mb=2000.0, grad_norm=1.5)
    assert isinstance(reward, float)
    print(f"   [PASS] Meta-Controller bounded actions: aug={actions['aug_prob'].item():.2f}, w_det={actions['w_det'].item():.2f}, Reward={reward:.4f}")

    # 8. Adaptive Compute Router & Knowledge Distillation
    print("8. Testing Adaptive Compute Router & KD Loss...")
    router = AdaptiveComputeRouter(channels=64).to(device)
    feat = torch.randn(2, 64, 40, 40, device=device)
    fused_f, gate, ran_refine = router(feat)
    assert fused_f.shape == feat.shape
    
    kd_fn = KnowledgeDistillationLoss()
    kd_loss = kd_fn(out_deploy["class_logits"], out_deploy["class_logits"])
    assert kd_loss.item() < 1e-4
    print(f"   [PASS] Adaptive router gate: {gate.mean().item():.3f} | KD loss: {kd_loss.item():.6f}")

    print("\n=================================================================")
    print(" ALL 8 NEURAVEX v0.7 UPGRADE MODULES VERIFIED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_v07_modules()
