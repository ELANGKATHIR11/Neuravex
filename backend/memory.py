import os
import json
import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("neuravex.memory")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MEMORY_FILE = os.path.join(DATA_DIR, "marked_entities.json")

class ObjectMemoryStore:
    """
    Persistent Real-time Object Memory & Identification Registry.
    Remembers objects by ID, RFID tag, nickname, classification, and physical 3D signatures.
    Persists marked objects to disk across runs.
    """
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.marked_registry: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    self.marked_registry = json.load(f)
                logger.info("Loaded %d marked objects from memory store", len(self.marked_registry))
            except Exception as e:
                logger.error("Failed to load marked entities memory: %s", e)
                self.marked_registry = {}

    def save(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.marked_registry, f, indent=2)
        except Exception as e:
            logger.error("Failed to save marked entities memory: %s", e)

    def mark_entity(
        self,
        entity_id: str,
        tag: Optional[str] = None,
        nickname: Optional[str] = None,
        notes: Optional[str] = None,
        entity_type: Optional[str] = None,
        position3D: Optional[Dict[str, float]] = None,
        dimensions3D: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Marks an entity to be remembered forever with its ID, custom tag, and nickname.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        existing = self.marked_registry.get(entity_id, {})

        record = {
            "id": entity_id,
            "tag": tag if tag is not None else existing.get("tag", f"ID-{entity_id}"),
            "nickname": nickname if nickname is not None else existing.get("nickname", ""),
            "notes": notes if notes is not None else existing.get("notes", ""),
            "type": entity_type or existing.get("type", "unknown"),
            "marked": True,
            "firstMarked": existing.get("firstMarked", now),
            "lastSeen": now,
            "lastKnownPosition": position3D or existing.get("lastKnownPosition"),
            "dimensions3D": dimensions3D or existing.get("dimensions3D")
        }

        self.marked_registry[entity_id] = record
        self.save()
        logger.info("Marked entity saved to memory: %s (%s)", entity_id, record["nickname"] or record["tag"])
        return record

    def unmark_entity(self, entity_id: str) -> bool:
        if entity_id in self.marked_registry:
            del self.marked_registry[entity_id]
            self.save()
            logger.info("Unmarked entity removed from memory: %s", entity_id)
            return True
        return False

    def is_marked(self, entity_id: str) -> bool:
        return entity_id in self.marked_registry

    def get_marked(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.marked_registry.get(entity_id)

    def get_all_marked(self) -> List[Dict[str, Any]]:
        return list(self.marked_registry.values())

    def enrich_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the entity matches a remembered marked object (by ID or tag),
        and enriches it with remembered metadata.
        """
        eid = entity.get("id")
        record = self.marked_registry.get(eid)

        if record:
            entity["marked"] = True
            entity["nickname"] = record.get("nickname", "")
            if record.get("tag"):
                entity["tag"] = record["tag"]
            if record.get("notes"):
                entity["notes"] = record["notes"]
            # Update last known position
            record["lastSeen"] = entity.get("lastSeen", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            if entity.get("position3D"):
                record["lastKnownPosition"] = entity["position3D"]
            if entity.get("dimensions3D"):
                record["dimensions3D"] = entity["dimensions3D"]
        else:
            entity["marked"] = False
            entity["nickname"] = ""
            entity["notes"] = ""

        return entity

# Global singleton memory store
memory_store = ObjectMemoryStore()
