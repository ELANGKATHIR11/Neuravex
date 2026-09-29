"""
Native Entity Marking, 3D Spatial Memory, and Persistent Identity Management.
"""

import time
import math
from typing import Dict, Any, Optional, List, Tuple
import torch
from .geometry3d import BoundingBox3D

class MarkedEntity:
    """
    Represents an explicitly marked entity in 3D spatial memory.
    """
    def __init__(
        self,
        entity_id: str,
        name: str,
        initial_box3d: BoundingBox3D,
        notes: str = "",
        category: str = "default",
        tags: Optional[List[str]] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.current_box3d = initial_box3d
        self.notes = notes
        self.category = category
        self.tags = tags or []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.trajectory_3d: List[Tuple[float, float, float]] = [
            (float(initial_box3d.center[0]), float(initial_box3d.center[1]), float(initial_box3d.center[2]))
        ]

    def update_location(self, new_box3d: BoundingBox3D):
        """Update entity spatial location and append to trajectory."""
        self.current_box3d = new_box3d
        self.updated_at = time.time()
        self.trajectory_3d.append(
            (float(new_box3d.center[0]), float(new_box3d.center[1]), float(new_box3d.center[2]))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert marked entity state to dictionary."""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "category": self.category,
            "notes": self.notes,
            "tags": self.tags,
            "current_3d": self.current_box3d.to_dict(),
            "trajectory_points": len(self.trajectory_3d),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class SpatialMarkerStore:
    """
    Persistent memory store for spatial entities and user markers.
    """
    def __init__(self):
        self.entities: Dict[str, MarkedEntity] = {}

    def mark_entity(
        self,
        entity_id: str,
        name: str,
        box3d: BoundingBox3D,
        notes: str = "",
        category: str = "marked"
    ) -> MarkedEntity:
        """Mark a new entity or update existing entity."""
        if entity_id in self.entities:
            ent = self.entities[entity_id]
            ent.name = name
            ent.notes = notes
            ent.update_location(box3d)
            return ent

        ent = MarkedEntity(entity_id=entity_id, name=name, initial_box3d=box3d, notes=notes, category=category)
        self.entities[entity_id] = ent
        return ent

    def remove_entity(self, entity_id: str) -> bool:
        """Delete an entity from memory."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False

    def get_entity(self, entity_id: str) -> Optional[MarkedEntity]:
        """Fetch entity by ID."""
        return self.entities.get(entity_id)

    def find_nearest(self, query_xyz: Tuple[float, float, float]) -> Optional[Tuple[MarkedEntity, float]]:
        """Find the closest marked entity to a query 3D location."""
        if not self.entities:
            return None

        q = torch.tensor(query_xyz, dtype=torch.float32)
        best_ent = None
        min_dist = float("inf")

        for ent in self.entities.values():
            dist = float(torch.norm(ent.current_box3d.center.cpu() - q).item())
            if dist < min_dist:
                min_dist = dist
                best_ent = ent

        return (best_ent, min_dist) if best_ent else None

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all marked entities in memory."""
        return [ent.to_dict() for ent in self.entities.values()]
