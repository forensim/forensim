"""
End-to-end reconstruction pipeline orchestrator.

Coordinates the three reconstruction stages:
    1. COLMAP Structure-from-Motion
    2. Gaussian Splatting (or fallback cloud export)
    3. USD export via omniverse-gsplat-converter

The orchestrator is responsible for:
    - Validating inputs and creating a deterministic workspace layout
    - Calling each stage with progress callbacks
    - Writing a JSON manifest describing the run
    - Providing a clean error boundary around the whole flow

Example:
    from pathlib import Path
    from forensim.reconstruct.pipeline import ReconstructionPipeline

    pipeline = ReconstructionPipeline()
    result = pipeline.run(
        image_dir=Path("evidence/images"),
        workspace_dir=Path("workspace/run_1"),
    )
    print(result.usd_path)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from forensim.reconstruct import colmap, gsplat, usd_export

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


@dataclass
class PipelineResult:
    """Result of a reconstruction pipeline run."""

    status: str
    workspace_dir: Path
    image_dir: Path
    sparse_dir: Path
    ply_path: Path
    usd_path: Path
    duration_seconds: float
    steps: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "workspace_dir": str(self.workspace_dir),
            "image_dir": str(self.image_dir),
            "sparse_dir": str(self.sparse_dir),
            "ply_path": str(self.ply_path),
            "usd_path": str(self.usd_path),
            "duration_seconds": self.duration_seconds,
            "steps": self.steps,
            "error": self.error,
        }


class ReconstructionPipeline:
    """
    Orchestrates image → COLMAP → Gaussian Splat → USD.

    Args:
        matcher: COLMAP matcher, either 'exhaustive' or 'sequential'.
        max_steps: Maximum training steps for real Gaussian Splatting trainers.
        resolution: Image downscale factor for real trainers.
        gsplat_fallback: If True, use the point-cloud Gaussian exporter when
                         a real trainer is not available.
        up_axis: USD scene up-axis ('Y' or 'Z').
    """

    def __init__(
        self,
        matcher: str = "exhaustive",
        max_steps: int = 30_000,
        resolution: int = -1,
        gsplat_fallback: bool = True,
        up_axis: str = "Y",
    ) -> None:
        self.matcher = matcher
        self.max_steps = max_steps
        self.resolution = resolution
        self.gsplat_fallback = gsplat_fallback
        self.up_axis = up_axis

    def _default_progress(self, step: str, percent: float, message: str) -> None:
        logger.info("[%-12s] %3.0f%% %s", step, percent, message)

    def _copy_images(self, image_dir: Path, workspace_dir: Path) -> Path:
        """Optionally copy images into the workspace for reproducibility."""
        # For now, operate on the original images to avoid duplicating evidence.
        return image_dir

    def _write_manifest(self, result: PipelineResult) -> Path:
        manifest_path = result.workspace_dir / "manifest.json"
        manifest_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info("Pipeline manifest written to %s", manifest_path)
        return manifest_path

    def run(
        self,
        image_dir: Path,
        workspace_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """
        Run the full reconstruction pipeline.

        Args:
            image_dir: Directory containing input images.
            workspace_dir: Directory where all outputs will be written.
            progress: Optional callback(step_name, percent, message).

        Returns:
            PipelineResult describing the run.
        """
        if progress is None:
            progress = self._default_progress

        start_time = time.perf_counter()
        steps: list[dict[str, object]] = []

        def _track(step: str, percent: float, message: str) -> None:
            steps.append({"step": step, "percent": percent, "message": str(message)})
            progress(step, percent, message)

        image_dir = Path(image_dir)
        workspace_dir = Path(workspace_dir)

        if not image_dir.exists():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")

        workspace_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir = workspace_dir / "sparse"
        gsplat_dir = workspace_dir / "gsplat"
        ply_path = gsplat_dir / "point_cloud.ply"
        usd_path = workspace_dir / "scene.usdz"

        try:
            _track("pipeline", 0.05, "Starting reconstruction pipeline")

            # Step 1: COLMAP SfM.
            _track("colmap", 0.0, "Running COLMAP Structure-from-Motion")
            sparse_dir = colmap.full_pipeline(
                image_dir=image_dir,
                workspace_dir=workspace_dir,
                matcher=self.matcher,
                progress=_track,
            )

            colmap_dir = sparse_dir / "0"
            if not (colmap_dir / "points3D.bin").exists():
                # COLMAP sometimes writes the model directly into sparse/ instead of sparse/0/
                if (sparse_dir / "points3D.bin").exists():
                    colmap_dir = sparse_dir
                else:
                    raise FileNotFoundError(
                        f"COLMAP did not produce points3D.bin in {sparse_dir}"
                    )

            _track("colmap", 1.0, "COLMAP Structure-from-Motion complete")

            # Step 2: Gaussian Splatting.
            _track("gsplat", 0.0, "Training Gaussian Splatting model")
            ply_path = gsplat.train(
                colmap_dir=colmap_dir,
                image_dir=image_dir,
                output_dir=gsplat_dir,
                max_steps=self.max_steps,
                resolution=self.resolution,
                fallback=self.gsplat_fallback,
                progress=_track,
            )
            _track("gsplat", 1.0, "Gaussian Splatting model ready")

            # Step 3: USD export.
            _track("usd", 0.0, "Exporting to USD")
            usd_path = usd_export.ply_to_usd(
                ply_path=ply_path,
                output_path=usd_path,
                up_axis=self.up_axis,
            )
            _track("usd", 1.0, "USD export complete")

            duration = time.perf_counter() - start_time
            result = PipelineResult(
                status="success",
                workspace_dir=workspace_dir,
                image_dir=image_dir,
                sparse_dir=sparse_dir,
                ply_path=ply_path,
                usd_path=usd_path,
                duration_seconds=duration,
                steps=steps,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.exception("Reconstruction pipeline failed")
            result = PipelineResult(
                status="failed",
                workspace_dir=workspace_dir,
                image_dir=image_dir,
                sparse_dir=sparse_dir,
                ply_path=ply_path,
                usd_path=usd_path,
                duration_seconds=duration,
                steps=steps,
                error=str(exc),
            )
            raise
        finally:
            self._write_manifest(result)

        return result

    def reset_workspace(self, workspace_dir: Path) -> None:
        """Delete and recreate the workspace directory."""
        workspace_dir = Path(workspace_dir)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
