"""
Native Spatial Flow & Volumetric Counting Engine:
Directional 3D boundaries, virtual tripwires, and volumetric density aggregation.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
from .geometry3d import BoundingBox3D

class CountingZone3D:
    """
    Volumetric 3D Counting Zone defined by min/max boundaries (X, Y, Z).
    """
    def __init__(
        self,
        zone_id: str,
        name: str,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float
    ):
        self.zone_id = zone_id
        self.name = name
        self.x_bounds = (float(x_min), float(x_max))
        self.y_bounds = (float(y_min), float(y_max))
        self.z_bounds = (float(z_min), float(z_max))
        self.entered_ids = set()
        self.total_count = 0

    def contains_point(self, point_xyz: torch.Tensor) -> bool:
        """Check if 3D point is inside the spatial zone."""
        x, y, z = float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])
        return (
            (self.x_bounds[0] <= x <= self.x_bounds[1]) and
            (self.y_bounds[0] <= y <= self.y_bounds[1]) and
            (self.z_bounds[0] <= z <= self.z_bounds[1])
        )

    def update(self, box: BoundingBox3D) -> bool:
        """
        Updates count if an object enters the zone.
        Returns True if a new entry was registered.
        """
        inside = self.contains_point(box.center)
        track_id = box.track_id or id(box)

        if inside and (track_id not in self.entered_ids):
            self.entered_ids.add(track_id)
            self.total_count += 1
            return True
        return False


class SpatialCounter:
    """
    Aggregates per-class spatial counts, directional tripwires, and volumetric zones.
    """
    def __init__(self):
        self.zones: Dict[str, CountingZone3D] = {}
        self.class_counts: Dict[str, int] = {}
        self.active_tracks: set = set()

    def add_zone(self, zone: CountingZone3D):
        self.zones[zone.zone_id] = zone

    def process_detections(self, boxes_3d: List[BoundingBox3D]) -> Dict[str, Any]:
        """
        Process a list of 3D detected boxes for zone counting and class accumulation.
        """
        current_frame_counts = {}
        for box in boxes_3d:
            cls = box.class_name
            current_frame_counts[cls] = current_frame_counts.get(cls, 0) + 1

            # Accumulate overall unique counts by track_id
            if box.track_id is not None:
                if box.track_id not in self.active_tracks:
                    self.active_tracks.add(box.track_id)
                    self.class_counts[cls] = self.class_counts.get(cls, 0) + 1

            # Check volumetric zones
            for zone in self.zones.values():
                zone.update(box)

        return {
            "current_frame_active": current_frame_counts,
            "cumulative_unique_counts": self.class_counts,
            "zone_counts": {zid: z.total_count for zid, z in self.zones.items()},
            "total_tracked_objects": len(self.active_tracks)
        }
