import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotAttention(nn.Module):
    """
    Slot Attention module for Unsupervised Object Discovery (Locatello et al.):
    Iteratively binds $K$ object slots to spatial feature clusters via competitive softmax cross-attention.
    Discovers distinct foreground objects, anomalies, and obstacles with zero human bounding-box labels.
    """
    def __init__(self, num_slots: int = 8, slot_dim: int = 128, in_features: int = 192,
                 iters: int = 3, eps: float = 1e-8, hidden_dim: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Learnable Gaussian slot initialization parameters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)

        # Projections
        self.project_k = nn.Linear(in_features, slot_dim, bias=False)
        self.project_v = nn.Linear(in_features, slot_dim, bias=False)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # Recurrent state update (GRUCell)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Residual MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim)
        )

        self.norm_inputs = nn.LayerNorm(in_features)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor, num_slots: int = None) -> tuple:
        """
        inputs: (B, N, C) spatial patch/token features where N = H * W
        num_slots: optional override for number of discovery slots
        Returns:
            slots: (B, K, slot_dim)
            attn: (B, K, N) attention assignment weights per slot
        """
        B, N, C = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        inputs = self.norm_inputs(inputs)

        k = self.project_k(inputs) # (B, N, slot_dim)
        v = self.project_v(inputs) # (B, N, slot_dim)

        # Initialize slots from Gaussian distribution
        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_logsigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)

        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            q = self.project_q(slots_norm) # (B, K, slot_dim)

            # Dot-product attention logits: (B, K, N)
            dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
            
            # Competitive attention normalized across SLOTS (dim 1)
            attn = F.softmax(dots, dim=1) + self.eps # (B, K, N)
            attn_norm = attn / torch.sum(attn, dim=-1, keepdim=True) # (B, K, N)

            # Weighted sum updates
            updates = torch.bmm(attn_norm, v) # (B, K, slot_dim)

            # Recurrent GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, K, self.slot_dim)

            # Residual MLP
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn

class UnsupervisedObjectDiscoveryHead(nn.Module):
    """
    High-level head binding Slot Attention to high-level multi-scale features (P5/Q5).
    Discovers:
      1. Spatial foreground object masks without labels
      2. 3D estimated center and spatial extent (bounding cuboids)
      3. Objectness/novelty confidence scores
    """
    def __init__(self, in_channels: int, num_slots: int = 8, slot_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Map convolutional features to slot input dimension
        self.conv_in = nn.Conv2d(in_channels, in_channels, 1)
        self.slot_attention = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, in_features=in_channels)

        # Slot decoder: decodes each slot into foreground object mask, 3D box, and objectness score
        self.mask_decoder = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1) # per-pixel slot mask weight
        )
        self.box_3d_head = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 6) # (cx, cy, cz, l, w, h)
        )
        self.objectness_head = nn.Sequential(
            nn.Linear(slot_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_map: torch.Tensor) -> dict:
        """
        feat_map: (B, C, H, W) high-level feature map (e.g. Q5 or P5)
        Returns:
            slots: (B, K, slot_dim)
            slot_masks: (B, K, H, W) spatial segmentation per object slot
            pred_3d_boxes: (B, K, 6) 3D bounding cuboids for discovered objects
            objectness: (B, K, 1) confidence that slot contains a valid physical entity
        """
        B, C, H, W = feat_map.shape
        x = self.conv_in(feat_map)
        x_flat = x.flatten(2).transpose(1, 2) # (B, H*W, C)

        slots, attn = self.slot_attention(x_flat) # slots: (B, K, D), attn: (B, K, H*W)

        # Spatial masks
        slot_masks = attn.reshape(B, self.num_slots, H, W)

        # 3D cuboids & objectness per slot
        pred_3d = self.box_3d_head(slots) # (B, K, 6)
        pred_xyz = pred_3d[..., :3]
        pred_lwh = torch.exp(pred_3d[..., 3:].clamp(-3.0, 3.0))
        pred_boxes_3d = torch.cat([pred_xyz, pred_lwh], dim=-1)

        objectness = self.objectness_head(slots) # (B, K, 1)

        return {
            "slots": slots,
            "slot_masks": slot_masks,
            "discovered_boxes_3d": pred_boxes_3d,
            "slot_objectness": objectness
        }
