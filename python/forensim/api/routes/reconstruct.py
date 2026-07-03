"""Reconstruction pipeline API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from forensim.reconstruct.pipeline import ReconstructionPipeline

router = APIRouter()


class ReconstructRequest(BaseModel):
    image_dir: str
    workspace_dir: str
    method: str = "gaussian_splatting"  # "gaussian_splatting" | "nerf" | "nurec"
    matcher: str = "exhaustive"          # "exhaustive" | "sequential"
    max_steps: int = 30_000
    gsplat_fallback: bool = True


class ReconstructResponse(BaseModel):
    status: str
    usd_path: str
    ply_path: str | None = None
    duration_seconds: float
    message: str | None = ""


@router.post("/run", response_model=ReconstructResponse)
async def run_reconstruction(req: ReconstructRequest) -> ReconstructResponse:
    """
    Run the full 2D → 3D → USD reconstruction pipeline.

    1. COLMAP SfM
    2. Gaussian Splatting (gsplat) or fallback point-cloud export
    3. Export to USD
    """
    image_dir = Path(req.image_dir)
    workspace_dir = Path(req.workspace_dir)

    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"image_dir not found: {image_dir}")

    try:
        pipeline = ReconstructionPipeline(
            matcher=req.matcher,
            max_steps=req.max_steps,
            gsplat_fallback=req.gsplat_fallback,
        )
        result = pipeline.run(image_dir=image_dir, workspace_dir=workspace_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReconstructResponse(
        status=result.status,
        usd_path=str(result.usd_path),
        ply_path=str(result.ply_path),
        duration_seconds=result.duration_seconds,
        message="Reconstruction pipeline complete" if result.status == "success" else result.error,
    )
