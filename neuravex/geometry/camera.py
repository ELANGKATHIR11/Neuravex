import torch

class CameraIntrinsics:
    """
    Pinhole camera model with parameters K = (fx, fy, cx, cy).
    Supports converting between 2D pixel coordinates + depth and 3D camera coordinates (X, Y, Z).
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float, device="cpu"):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.device = device

    def scale(self, scale_x: float, scale_y: float) -> "CameraIntrinsics":
        """
        Adjust intrinsics for isotropic or anisotropic image resizing:
          fx' = fx * sx, fy' = fy * sy
          cx' = cx * sx, cy' = cy * sy
        """
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            device=self.device
        )

    def adjust_letterbox(self, scale: float, pad_x: float, pad_y: float) -> "CameraIntrinsics":
        """
        Adjust intrinsics after image letterboxing (scaling by `scale` followed by padding):
          fx' = fx * scale, fy' = fy * scale
          cx' = cx * scale + pad_x, cy' = cy * scale + pad_y
        """
        return CameraIntrinsics(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale + pad_x,
            cy=self.cy * scale + pad_y,
            device=self.device
        )

    def unproject_points(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Unprojects (u, v) image coordinates and depth Z into camera coordinates (X, Y, Z):
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = Z
        Returns (..., 3).
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return torch.stack([x, y, z], dim=-1)

    def project_points(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Projects (X, Y, Z) in camera frame to pixel coordinates (u, v):
            u = X * fx / Z + cx
            v = Y * fy / Z + cy
        Returns (..., 2).
        """
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2].clamp(min=1e-4)
        u = x * self.fx / z + self.cx
        v = y * self.fy / z + self.cy
        return torch.stack([u, v], dim=-1)

    def unproject_depth_map(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Converts a full dense depth map (B, 1, H, W) to dense 3D point map (B, 3, H, W).
        """
        B, _, H, W = depth.shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=depth.dtype),
            torch.arange(W, device=depth.device, dtype=depth.dtype),
            indexing="ij"
        )
        u = x_grid.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        v = y_grid.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return torch.cat([x, y, z], dim=1)

def depth_to_inverse(depth: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Computes inverse depth rho = 1 / (Z + eps)."""
    return 1.0 / (depth.clamp_min(eps))

def inverse_to_depth(rho: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Recovers depth Z = 1 / (rho + eps)."""
    return 1.0 / (rho.clamp_min(eps))
