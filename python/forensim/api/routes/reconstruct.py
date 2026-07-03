"""Reconstruction pipeline API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ReconstructRequest(BaseModel):
    image_dir: str
    workspace_dir: str
    method: str = "gaussian_splatting"  # "gaussian_splatting" | "nerf" | "nurec"
    matcher: str = "exhaustive"          # "exhaustive" | "sequential"
    max_steps: int = 30_000


class ReconstructResponse(BaseModel):
    status: str
    usd_path: str
    ply_path: str | None = None
    message: str = ""


@router.post("/run", response_model=ReconstructResponse)
async def run_reconstruction(req: ReconstructRequest) -> ReconstructResponse:
    """
    Run the full 2D → 3D → USD reconstruction pipeline.

    1. COLMAP SfM
    2. Gaussian Splatting (gsplat) or NeRF
    3. Export to USD
    """
    image_dir = Path(req.image_dir)
    workspace_dir = Path(req.workspace_dir)

    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"image_dir not found: {image_dir}")

    try:
        from forensim.reconstruct import colmap, gsplat, usd_export

        # Step 1: COLMAP
        sparse_dir = colmap.full_pipeline(image_dir, workspace_dir, matcher=req.matcher)

        ply_path: Path | None = None
        usd_path: Path | None = None

        if req.method == "gaussian_splatting":
            # Step 2: Gaussian Splatting
            colmap_dir = sparse_dir / "0"
            ply_path = gsplat.train(
                colmap_dir=colmap_dir,
                image_dir=image_dir,
                output_dir=workspace_dir / "gsplat",
                max_steps=req.max_steps,
            )
            # Step 3: USD export
            usd_path = usd_export.ply_to_usd(
                ply_path=ply_path,
                output_path=workspace_dir / "scene.usdz",
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

        return ReconstructResponse(
            status="success",
            usd_path=str(usd_path),
            ply_path=str(ply_path) if ply_path else None,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
