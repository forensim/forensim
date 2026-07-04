"""Load and persist evidence annotations as JSON sidecars."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Shape(StrEnum):
    """Supported annotation ROI shapes."""

    RECT = "rect"
    POLYGON = "polygon"


@dataclass
class Annotation:
    """A single region-of-interest annotation on an evidence image."""

    id: str
    image_path: str
    shape: Shape
    coordinates: list[list[float]]
    """Rectangle: [[x1, y1], [x2, y2]]. Polygon: [[x1, y1], [x2, y2], ...]."""
    tag: str
    description: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.shape, str):
            self.shape = Shape(self.shape)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = self.shape.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        return cls(
            id=data["id"],
            image_path=data["image_path"],
            shape=data["shape"],
            coordinates=data["coordinates"],
            tag=data["tag"],
            description=data.get("description", ""),
            confidence=float(data.get("confidence", 1.0)),
            metadata=data.get("metadata", {}),
        )


class AnnotationManager:
    """Save and load annotations for a workspace."""

    FILENAME = "annotations.json"

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.annotations: list[Annotation] = []

    def _path(self) -> Path:
        return self.workspace_dir / self.FILENAME

    def load(self) -> list[Annotation]:
        """Load annotations from the workspace JSON sidecar."""
        path = self._path()
        if not path.exists():
            self.annotations = []
            return self.annotations

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse annotation file %s: %s", path, exc)
            self.annotations = []
        return self.annotations

    def save(self, annotations: list[Annotation] | None = None) -> Path:
        """Persist annotations to the workspace JSON sidecar."""
        if annotations is not None:
            self.annotations = annotations

        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self._path()
        payload = {
            "version": 1,
            "annotations": [a.to_dict() for a in self.annotations],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved %d annotations to %s", len(self.annotations), path)
        return path

    def add(self, annotation: Annotation) -> None:
        """Add a new annotation and save."""
        self.annotations.append(annotation)
        self.save()

    def delete(self, annotation_id: str) -> bool:
        """Delete an annotation by id and save."""
        original_len = len(self.annotations)
        self.annotations = [a for a in self.annotations if a.id != annotation_id]
        if len(self.annotations) != original_len:
            self.save()
            return True
        return False

    def clear(self) -> None:
        """Remove all annotations and save."""
        self.annotations = []
        self.save()
