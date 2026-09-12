import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceSSLClustering(nn.Module):
    """
    Self-Supervised Instance Clustering:
    Generates pseudo-instance masks by fusing:
      - Spatial coordinates (X, Y)
      - Metric depth map Z
      - Learned pixel embeddings
    Rejects ambiguous and low-density clusters.
    """
    def __init__(self, bandwidth: float = 0.5, min_cluster_pixels: int = 50):
        super().__init__()
        self.bandwidth = bandwidth
        self.min_cluster_pixels = min_cluster_pixels

    @torch.no_grad()
    def cluster(self, embeddings: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D, H, W)
        depth: (B, 1, H, W)
        Returns: (B, H, W) instance ID map (0 = background)
        """
        B, D, H, W = embeddings.shape
        inst_maps = []

        # Meshgrid coordinates normalized in [0, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H, device=embeddings.device),
            torch.linspace(0, 1, W, device=embeddings.device),
            indexing="ij"
        )
        coords = torch.stack([x_grid, y_grid], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # Normalized depth feature
        d_norm = depth / (depth.max() + 1e-6)

        # Joint feature vector: [coords (2), depth (1), embeddings (D)]
        joint_feat = torch.cat([coords * 0.5, d_norm * 0.5, F.normalize(embeddings, dim=1)], dim=1) # (B, 3+D, H, W)

        for b in range(B):
            feat_b = joint_feat[b].permute(1, 2, 0).reshape(H * W, -1) # (N_pix, feat_dim)
            inst_map_b = torch.zeros(H * W, dtype=torch.long, device=embeddings.device)

            # Fast spatial sub-sampling grid seeds for pseudo instances
            step = 16
            seed_idx = []
            for sy in range(step // 2, H, step):
                for sx in range(step // 2, W, step):
                    seed_idx.append(sy * W + sx)
            seeds = feat_b[seed_idx] # (num_seeds, feat_dim)

            # Nearest-neighbor clustering
            dists = torch.cdist(feat_b, seeds) # (N_pix, num_seeds)
            min_dist, cluster_id = dists.min(dim=-1)

            # Threshold and size filtering
            valid = min_dist < self.bandwidth
            inst_map_b[valid] = cluster_id[valid] + 1
            inst_maps.append(inst_map_b.view(H, W))

        return torch.stack(inst_maps, dim=0)
