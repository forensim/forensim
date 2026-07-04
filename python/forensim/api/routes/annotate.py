"""Annotation API routes: save/load evidence ROI annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from forensim.annotate.manager import Annotation, AnnotationManager, Shape

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────


class AnnotationModel(BaseModel):
    id: str
    image_path: str
    shape: str
    coordinates: list[list[float]]
    tag: str
    description: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = {}


class SaveAnnotationsRequest(BaseModel):
    workspace_dir: str
    annotations: list[AnnotationModel]


class SaveAnnotationsResponse(BaseModel):
    status: str
    saved_path: str
    count: int


class LoadAnnotationsResponse(BaseModel):
    status: str
    annotations: list[AnnotationModel]


# ── Routes ────────────────────────────────────────────────────────────────────


def _to_annotation_model(a: Annotation) -> AnnotationModel:
    return AnnotationModel(
        id=a.id,
        image_path=a.image_path,
        shape=a.shape.value,
        coordinates=a.coordinates,
        tag=a.tag,
        description=a.description,
        confidence=a.confidence,
        metadata=a.metadata,
    )


@router.post("/save", response_model=SaveAnnotationsResponse)
async def save_annotations(req: SaveAnnotationsRequest) -> SaveAnnotationsResponse:
    """Persist annotations to a workspace JSON sidecar."""
    workspace_dir = Path(req.workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    annotations = [
        Annotation(
            id=a.id,
            image_path=a.image_path,
            shape=Shape(a.shape),
            coordinates=a.coordinates,
            tag=a.tag,
            description=a.description,
            confidence=a.confidence,
            metadata=a.metadata,
        )
        for a in req.annotations
    ]

    manager = AnnotationManager(workspace_dir)
    try:
        saved_path = manager.save(annotations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SaveAnnotationsResponse(
        status="success",
        saved_path=str(saved_path),
        count=len(annotations),
    )


@router.get("/load", response_model=LoadAnnotationsResponse)
async def load_annotations(workspace_dir: str) -> LoadAnnotationsResponse:
    """Load annotations from a workspace JSON sidecar."""
    path = Path(workspace_dir)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Workspace not found: {path}")

    manager = AnnotationManager(path)
    try:
        annotations = manager.load()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return LoadAnnotationsResponse(
        status="success",
        annotations=[_to_annotation_model(a) for a in annotations],
    )
