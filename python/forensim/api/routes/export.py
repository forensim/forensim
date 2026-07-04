"""Export API routes: PDF reports, USD packaging, and flythrough video."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from forensim.export.report import ReportInputs, generate_pdf_report
from forensim.export.usd import package_usd_scene
from forensim.export.video import generate_flythrough_video

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────


class ReportRequest(BaseModel):
    case_title: str
    examiner: str
    notes: str = ""
    output_path: str
    reconstruction: dict[str, Any] | None = None
    simulation: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    screenshot_bytes: list[str] | None = None
    """Base64-encoded PNG screenshots (optional)."""


class ReportResponse(BaseModel):
    status: str
    output_path: str


class UsdExportRequest(BaseModel):
    usd_path: str
    output_path: str


class UsdExportResponse(BaseModel):
    status: str
    output_path: str


class VideoRequest(BaseModel):
    ply_path: str | None = None
    trajectories: list[dict[str, Any]] = []
    output_path: str
    duration_seconds: float = 5.0
    fps: int = 30


class VideoResponse(BaseModel):
    status: str
    output_path: str


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/report", response_model=ReportResponse)
async def create_report(req: ReportRequest) -> ReportResponse:
    """Generate a PDF forensic report."""
    output_path = Path(req.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    screenshots: list[bytes] | None = None
    if req.screenshot_bytes:
        import base64

        screenshots = [base64.b64decode(b) for b in req.screenshot_bytes]

    inputs = ReportInputs(
        case_title=req.case_title,
        examiner=req.examiner,
        notes=req.notes,
        reconstruction=req.reconstruction,
        simulation=req.simulation,
        inference=req.inference,
        screenshot_bytes=screenshots,
    )

    try:
        out = await asyncio.get_event_loop().run_in_executor(
            None, generate_pdf_report, inputs, output_path
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ReportResponse(status="success", output_path=str(out))


@router.post("/usd", response_model=UsdExportResponse)
async def export_usd(req: UsdExportRequest) -> UsdExportResponse:
    """Package a USD scene and sibling assets into a zip archive."""
    usd_path = Path(req.usd_path)
    output_path = Path(req.output_path)

    if not usd_path.exists():
        raise HTTPException(status_code=400, detail=f"USD file not found: {usd_path}")

    try:
        out = await asyncio.get_event_loop().run_in_executor(
            None, package_usd_scene, usd_path, output_path
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UsdExportResponse(status="success", output_path=str(out))


@router.post("/video", response_model=VideoResponse)
async def create_video(req: VideoRequest) -> VideoResponse:
    """Generate an MP4 flythrough video from trajectories and/or a point cloud."""
    output_path = Path(req.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ply_path = Path(req.ply_path) if req.ply_path else None

    try:
        out = await asyncio.get_event_loop().run_in_executor(
            None,
            generate_flythrough_video,
            output_path,
            req.trajectories,
            ply_path,
            req.duration_seconds,
            req.fps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return VideoResponse(status="success", output_path=str(out))


@router.get("/download")
async def download_file(path: str) -> FileResponse:
    """Download an exported file by absolute path."""
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(
        str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )
