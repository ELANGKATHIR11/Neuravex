import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neuravex.engine.adaptive_compute import ActorCriticComputePolicy, compute_dynamic_budget_reward
from neuravex.losses.physics_rl import PhysicsConstraintReward, PhysicsTestTimeSelfCorrector


def test_actor_critic_compute_policy():
    policy = ActorCriticComputePolicy(state_dim=32, hidden_dim=64, num_actions=3)
    state = torch.randn(4, 32, 8, 8)

    # 1. Forward action sampling
    action, log_prob, value, entropy = policy(state)
    assert action.shape == (4,)
    assert (action >= 0).all() and (action < 3).all()
    assert log_prob.shape == (4,)
    assert value.shape == (4, 1)

    # 2. PPO Loss evaluation
    new_log_prob, new_val, new_ent = policy.evaluate_actions(state, action)
    returns = torch.ones(4)
    advantages = torch.ones(4)
    rl_loss = policy.compute_rl_loss(log_prob, new_log_prob, new_val, returns, advantages, new_ent)
    assert "loss_rl_total" in rl_loss
    assert not torch.isnan(rl_loss["loss_rl_total"])

def test_dynamic_budget_reward():
    iou = torch.tensor([0.8, 0.5])
    conf = torch.tensor([0.9, 0.4])
    action = torch.tensor([0, 2])
    target_lost = torch.tensor([0, 1])

    reward = compute_dynamic_budget_reward(iou, conf, action, target_lost)
    assert reward.shape == (2,)
    # Sample 0 had high accuracy and action 0 (low cost) -> positive reward
    # Sample 1 had target lost -> negative reward
    assert reward[0] > reward[1]

def test_physics_constraint_and_corrector():
    reward_fn = PhysicsConstraintReward(terrain_margin=0.2)
    corrector = PhysicsTestTimeSelfCorrector()

    # 2 overlapping boxes (collision)
    # [x, y, z, l, w, h, yaw]
    boxes = torch.tensor([
        [0.0, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0],
        [0.2, 0.0, 5.0, 1.0, 1.0, 1.0, 0.0]
    ])

    res = reward_fn(boxes)
    assert "reward_physics" in res
    assert res["penalty_collision"] > 0.0

    refined = corrector.correct_3d_boxes(boxes)
    assert refined.shape == (2, 7)
    # Collision separation check: distance should increase
    orig_dist = (boxes[1, :3] - boxes[0, :3]).norm()
    refined_dist = (refined[1, :3] - refined[0, :3]).norm()
    assert refined_dist > orig_dist
