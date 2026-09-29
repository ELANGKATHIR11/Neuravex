"""
Native Digital Elevation Model (DEM): Surface elevation maps,
terrain slope, aspect calculation, and ground-plane plane fitting.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F

class NativeDEMSurface:
    """
    Digital Elevation Model representation converting 2D metric depth maps
    into a calibrated geographical / metric terrain surface.
    """
    def __init__(self, elevation_grid: torch.Tensor, cell_resolution_m: float = 0.05):
        """
        Args:
            elevation_grid: (H, W) tensor of surface elevation in meters
            cell_resolution_m: physical distance between adjacent pixels (e.g. 5cm/pixel)
        """
        self.grid = elevation_grid.float()
        self.res = float(cell_resolution_m)
        self.device = elevation_grid.device

    @property
    def min_elevation(self) -> float:
        return float(self.grid.min().item())

    @property
    def max_elevation(self) -> float:
        return float(self.grid.max().item())

    @property
    def mean_elevation(self) -> float:
        return float(self.grid.mean().item())

    def compute_slopes_and_aspect(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes terrain slope (degrees) and aspect (orientation in degrees)
        using central finite differences.
        """
        grid = self.grid
        while grid.ndim > 2:
            grid = grid.squeeze(0)
        grid = grid.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        
        # Sobel gradient filters for elevation gradient dz/dx, dz/dy
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / (8.0 * self.res)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / (8.0 * self.res)

        dz_dx = F.conv2d(grid, sobel_x, padding=1).squeeze()
        dz_dy = F.conv2d(grid, sobel_y, padding=1).squeeze()

        # Slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        grad_mag = torch.sqrt(dz_dx**2 + dz_dy**2)
        slope_rad = torch.atan(grad_mag)
        slope_deg = torch.rad2deg(slope_rad)

        # Aspect = 180.0 - arctan2(dz/dy, -dz/dx)
        aspect_deg = torch.rad2deg(torch.atan2(dz_dy, -dz_dx)) % 360.0

        return slope_deg, aspect_deg

    def fit_ground_plane(self, quantile: float = 0.10) -> Tuple[torch.Tensor, float]:
        """
        Fits a RANSAC/least-squares ground plane equation: aX + bY + cZ + d = 0
        from the lower elevation quantiles of the DEM grid.
        """
        flat = self.grid.flatten()
        threshold = torch.quantile(flat, quantile)
        ground_mask = self.grid <= threshold

        v_idx, u_idx = torch.where(ground_mask)
        z_vals = self.grid[ground_mask]
        
        # Linear system [X, Y, 1] * [a, b, d]^T = Z
        x_m = u_idx.float() * self.res
        y_m = v_idx.float() * self.res

        A = torch.stack([x_m, y_m, torch.ones_like(x_m)], dim=-1)
        # Solve least squares
        sol = torch.linalg.lstsq(A, z_vals.unsqueeze(-1)).solution.squeeze()
        normal = torch.tensor([sol[0], sol[1], -1.0], device=self.device)
        normal = normal / torch.norm(normal)
        d_offset = float(sol[2].item())

        return normal, d_offset


def compute_elevation_profile(
    dem: NativeDEMSurface,
    start_pt: Tuple[int, int],
    end_pt: Tuple[int, int],
    num_samples: int = 100
) -> torch.Tensor:
    """
    Extracts a 1D elevation cross-section line between two 2D points on the DEM surface.
    """
    u0, v0 = start_pt
    u1, v1 = end_pt

    t = torch.linspace(0.0, 1.0, num_samples, device=dem.device)
    u_line = torch.clamp((u0 + t * (u1 - u0)).long(), 0, dem.grid.shape[1] - 1)
    v_line = torch.clamp((v0 + t * (v1 - v0)).long(), 0, dem.grid.shape[0] - 1)

    return dem.grid[v_line, u_line]
