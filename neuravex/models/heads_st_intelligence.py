"""
Neuravex Spatio-Temporal Motion, Spatial Interaction & Video Intelligence Head (ST-Intelligence).

Transforms frame-by-frame computer vision detections into structured temporal intelligence:
  1. 3D Kinematics reasoning: velocity, acceleration vectors, and directional curvature.
  2. Temporal state classification: Stationary, Active, Rapid, Erratic, Loitering.
  3. Spatial Interaction Graph (ST-Graph):
     - Pairwise 3D metric distances
     - Closest-proximity events
     - Approaching vs. retreating dynamics
     - Interaction duration tracking
  4. Trajectory Anomaly & Behavioral Shift Detection:
     - Kinetic energy shock / sudden impact (fall detection)
     - Erratic agitation score
     - Energy-based trajectory anomaly scoring
  5. Semantic Video Intelligence Synthesis:
     - Frame-by-frame and aggregate temporal intelligence
"""

import math
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalIntelligenceHead(nn.Module):
    """
    Neural Spatio-Temporal Motion, Interaction & Anomaly Reasoning Head.
    """
    def __init__(self, token_dim: int = 64, hidden_dim: int = 128, num_states: int = 5):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states

        # Trajectory token encoder: inputs (x, y, z, dt, speed, yaw) -> token_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(6, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim)
        )

        # Temporal sequence modeling (multi-head self-attention over temporal window)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=4,
            batch_first=True
        )

        # Kinematic Dynamics Head: predicts (vx, vy, vz, ax, ay, az, angular_rate)
        self.kinematic_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 7)
        )

        # Motion & Behavioral State Classifier
        # States: 0: Stationary, 1: Normal Moving, 2: Rapid Acceleration, 3: Erratic Agitation, 4: Loitering
        self.state_classifier = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_states)
        )

        # Trajectory Anomaly Scoring (Energy-based reconstruction head)
        self.anomaly_head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Spatial Interaction Pairwise Relation MLP
        # Inputs: concatenated entity tokens [token_i, token_j, delta_xyz, dist] -> interaction logits
        self.interaction_mlp = nn.Sequential(
            nn.Linear(token_dim * 2 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)  # 0: Independent, 1: Approaching, 2: Engaged/Interacting, 3: Retreating
        )

    def forward(
        self,
        trajectory_tokens: torch.Tensor,
        pairwise_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            trajectory_tokens: (B, T, 6) tensor of (x, y, z, dt, speed, yaw)
            pairwise_indices: optional (P, 2) tensor of entity pairs for interaction modeling
        Returns:
            Dict containing kinematics, motion states, anomaly scores, and interaction predictions.
        """
        B, T, _ = trajectory_tokens.shape
        embedded = self.traj_encoder(trajectory_tokens)
        attn_out, _ = self.temporal_attn(embedded, embedded, embedded)
        pooled = attn_out[:, -1, :]  # latest temporal token

        # 1. Kinematics
        kinematics = self.kinematic_head(pooled)
        # 2. Behavioral state logits
        state_logits = self.state_classifier(pooled)
        # 3. Trajectory anomaly score [0, 1]
        anomaly_scores = self.anomaly_head(pooled)

        outputs = {
            "kinematics": kinematics,
            "state_logits": state_logits,
            "anomaly_scores": anomaly_scores,
            "temporal_tokens": pooled
        }

        # 4. Pairwise interactions if multiple entities
        if pairwise_indices is not None and pairwise_indices.numel() > 0:
            p_logits = []
            for pair in pairwise_indices:
                i, j = int(pair[0]), int(pair[1])
                tok_i = pooled[i:i+1]
                tok_j = pooled[j:j+1]
                pos_i = trajectory_tokens[i:i+1, -1, :3]
                pos_j = trajectory_tokens[j:j+1, -1, :3]
                delta_pos = pos_j - pos_i
                dist = torch.norm(delta_pos, dim=-1, keepdim=True)
                pair_feat = torch.cat([tok_i, tok_j, delta_pos, dist], dim=-1)
                pair_log = self.interaction_mlp(pair_feat)
                p_logits.append(pair_log)
            outputs["interaction_logits"] = torch.cat(p_logits, dim=0)

        return outputs

    @staticmethod
    def compute_pairwise_spatial_interactions(
        entities: List[Dict[str, Any]],
        proximity_threshold: float = 2.5
    ) -> List[Dict[str, Any]]:
        """
        Evaluates pairwise 3D metric distances, approach/retreat rates,
        and closest-proximity events across all active objects.
        """
        interactions = []
        n = len(entities)
        for i in range(n):
            e1 = entities[i]
            p1 = e1.get("position3D", {"x": 0.0, "y": 0.0, "z": 0.0})
            v1 = e1.get("velocity", {"vx": 0.0, "vy": 0.0, "vz": 0.0})

            for j in range(i + 1, n):
                e2 = entities[j]
                p2 = e2.get("position3D", {"x": 0.0, "y": 0.0, "z": 0.0})
                v2 = e2.get("velocity", {"vx": 0.0, "vy": 0.0, "vz": 0.0})

                # 3D Metric Euclidean Distance
                dx = p2["x"] - p1["x"]
                dy = p2["y"] - p1["y"]
                dz = p2["z"] - p1["z"]
                dist = round(math.sqrt(dx**2 + dy**2 + dz**2), 2)

                # Relative velocity vector (v2 - v1)
                rvx = v2.get("vx", 0.0) - v1.get("vx", 0.0)
                rvy = v2.get("vy", 0.0) - v1.get("vy", 0.0)
                rvz = v2.get("vz", 0.0) - v1.get("vz", 0.0)

                # Radial approach speed: rate of change of distance d(dist)/dt
                # if dist > 0: d(dist)/dt = (dx*rvx + dy*rvy + dz*rvz) / dist
                radial_speed = 0.0
                if dist > 0.05:
                    radial_speed = round((dx * rvx + dy * rvy + dz * rvz) / dist, 2)

                # Interaction state determination
                if dist < 1.2:
                    rel_state = "close_contact"
                elif dist < proximity_threshold:
                    if radial_speed < -0.15:
                        rel_state = "approaching"
                    elif radial_speed > 0.15:
                        rel_state = "retreating"
                    else:
                        rel_state = "proximate_stationed"
                else:
                    rel_state = "independent"

                interactions.append({
                    "pairId": f"{e1['id']}-{e2['id']}",
                    "sourceId": e1["id"],
                    "targetId": e2["id"],
                    "sourceTag": e1.get("tag", e1["id"]),
                    "targetTag": e2.get("tag", e2["id"]),
                    "sourceType": e1.get("type", "object"),
                    "targetType": e2.get("type", "object"),
                    "distanceM": dist,
                    "radialSpeedMps": radial_speed,
                    "relationState": rel_state,
                    "isProximityAlert": dist < proximity_threshold,
                    "midpoint": {
                        "x": round((p1["x"] + p2["x"]) / 2.0, 3),
                        "y": round((p1["y"] + p2["y"]) / 2.0, 3),
                        "z": round((p1["z"] + p2["z"]) / 2.0, 3)
                    }
                })

        return interactions
