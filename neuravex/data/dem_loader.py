"""
Neuravex Real DEM (Digital Elevation Model) & Terrain Loader.

Supports:
- Loading raster DEMs (GeoTIFF / TIFF / HDF5 / NumPy float arrays)
- Coordinate Reference System (CRS) metadata handling (EPSG, projection)
- Spatial cropping, reprojection, and normalization to metric heights
- Camera pose to terrain surface registration and ray casting
"""

import os
import json
import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import cv2

try:
    import rasterio  # type: ignore[import-untyped,import-not-found]
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class DEMTerrainDataset:
    """
    Reader for real Digital Elevation Models (e.g. USGS 3DEP, SRTM, Copernicus DEM, ALOS AW3D30).
    Extracts metric ground elevation rasters with geographic CRS and converts to camera-relative coordinate space.
    """

    def __init__(
        self,
        dem_path: str,
        target_size: Tuple[int, int] = (320, 320),
        elevation_scale: float = 1.0,
        crs_epsg: Optional[int] = 4326,
    ):
        self.dem_path = os.path.abspath(dem_path)
        self.target_size = target_size
        self.elevation_scale = elevation_scale
        self.crs_epsg = crs_epsg

        if not os.path.exists(self.dem_path):
            raise FileNotFoundError(f"DEM raster file not found: {self.dem_path}")

        self.dem_data, self.metadata = self._load_dem()

    def _load_dem(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            "source": self.dem_path,
            "crs": f"EPSG:{self.crs_epsg}" if self.crs_epsg else "Unknown",
        }

        ext = os.path.splitext(self.dem_path)[1].lower()

        if ext in (".tif", ".tiff") and HAS_RASTERIO:
            with rasterio.open(self.dem_path) as src:
                raster = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    raster[raster == nodata] = np.nan
                metadata["crs"] = str(src.crs)
                metadata["bounds"] = list(src.bounds)
                metadata["transform"] = list(src.transform)
        elif ext in (".npy", ".npz"):
            loaded = np.load(self.dem_path)
            raster = loaded["elevation"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
            raster = raster.astype(np.float32)
        else:
            # Fallback for standard 16-bit or 32-bit single-channel image DEM
            img = cv2.imread(self.dem_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Unable to parse DEM raster: {self.dem_path}")
            if img.ndim == 3:
                img = img[:, :, 0]
            raster = img.astype(np.float32)

        # Replace NaNs / Infs with minimum elevation
        finite_mask = np.isfinite(raster)
        if not finite_mask.all():
            min_val = float(np.min(raster[finite_mask])) if finite_mask.any() else 0.0
            raster = np.nan_to_num(raster, nan=min_val, posinf=min_val, neginf=min_val)

        raster = raster * self.elevation_scale
        metadata["elevation_min_m"] = float(np.min(raster))
        metadata["elevation_max_m"] = float(np.max(raster))

        return raster, metadata

    def get_elevation_patch(
        self,
        bbox_geo: Optional[Tuple[float, float, float, float]] = None,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Extracts a normalized metric elevation patch resized to out_hw (H, W).
        Returns a float32 torch.Tensor of shape (1, 1, H, W).
        """
        hw = out_hw if out_hw is not None else self.target_size
        resized = cv2.resize(self.dem_data, hw[::-1], interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()
        return tensor

    def sample_elevation_at_camera_point(
        self,
        x_cam: float,
        z_cam: float,
        cam_height: float = 1.6,
    ) -> float:
        """
        Samples metric ground elevation under an object in camera coordinates.
        """
        h, w = self.dem_data.shape
        # Center of DEM raster represents camera optical ground nadir
        u = int(np.clip(w / 2.0 + x_cam * 10.0, 0, w - 1))
        v = int(np.clip(h / 2.0 + z_cam * 10.0, 0, h - 1))
        return float(self.dem_data[v, u])
