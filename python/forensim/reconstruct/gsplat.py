"""
3D Gaussian Splatting pipeline.

Provides a training entry point that can either delegate to the
``gaussian-splatting`` trainer (when CUDA is compiled and available)
or fall back to a deterministic point-cloud-to-Gaussian exporter that
creates a valid ``.ply`` file from COLMAP sparse points.

Strategy selection (checked in order):
    1. ``fallback=False`` + CUDA available → runs ``gaussian_splatting.train``
       with the ``gsplat`` backend.  Requires a compiled gsplat CUDA
       extension (JIT-compiled on first run when ``CUDA_HOME`` is set).
    2. ``fallback=True`` (default) → exports Gaussians directly from the
       COLMAP sparse point cloud via ``gsplat.export_splats``.  No CUDA
       compilation needed; works on any torch+CUDA build.

The fallback is the default because it does not require nvcc and lets
the COLMAP → USD pipeline be verified end-to-end immediately.  Switch to
``fallback=False`` (or set ``FORENSIM_GSPLAT_FALLBACK=0``) once the
CUDA Toolkit is installed and ``CUDA_HOME`` points to it.
"""

from __future__ import annotations

import logging
import os
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_colmap_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read COLMAP's binary points3D.bin file.

    Returns:
        (xyz, rgb): arrays of shape (N, 3). xyz is float64, rgb is uint8.
    """
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    num_points = struct.unpack_from("<Q", data, offset)[0]
    offset += 8

    xyz = np.empty((num_points, 3), dtype=np.float64)
    rgb = np.empty((num_points, 3), dtype=np.uint8)

    for i in range(num_points):
        # Format per point: Q ddd BBB d Q + (ii)*track_length
        offset += 8  # point3D_id
        xyz[i] = struct.unpack_from("<3d", data, offset)
        offset += 24
        rgb[i] = struct.unpack_from("<3B", data, offset)
        offset += 3
        offset += 8  # error (double)
        track_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        offset += 8 * track_length  # track entries: image_id (i32) + point2D_idx (i32)

    return xyz, rgb


# ---------------------------------------------------------------------------
# Fallback exporter: COLMAP sparse points → Gaussian PLY
# ---------------------------------------------------------------------------

def _export_splats_from_points(
    points_path: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Convert a COLMAP points3D.bin into a Gaussian Splat PLY.

    Each input point becomes one Gaussian with small scale and color
    derived from the COLMAP point color.
    """
    if progress:
        progress("gsplat", 0.0, "Reading COLMAP sparse point cloud...")

    xyz, rgb = _read_colmap_points3d_bin(points_path)
    if len(xyz) == 0:
        raise RuntimeError(f"COLMAP sparse point cloud is empty: {points_path}")

    logger.info("Loaded %d sparse points from COLMAP", len(xyz))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    means = torch.from_numpy(xyz).float().to(device)
    colors = torch.from_numpy(rgb).float().to(device) / 255.0

    num_points = means.shape[0]
    # Estimate a scene scale from point extents to set a reasonable Gaussian size.
    extents = xyz.max(axis=0) - xyz.min(axis=0)
    scene_scale = (
        float(np.percentile(extents[extents > 0], 50)) if np.any(extents > 0) else 1.0
    )
    base_scale = max(scene_scale * 0.02, 0.001)

    scales = torch.full((num_points, 3), base_scale, device=device)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(num_points, 1)
    opacities = torch.full((num_points,), 0.8, device=device)

    # sh0 is DC color, shN is higher-order SH (zero here).
    sh0 = colors.unsqueeze(1)  # (N, 1, 3)
    shN = torch.zeros((num_points, 0, 3), device=device)

    if progress:
        progress("gsplat", 0.5, f"Exporting {num_points} Gaussians to PLY...")

    from gsplat import export_splats  # type: ignore[import-untyped]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(output_path),
    )

    if progress:
        progress("gsplat", 1.0, f"Saved PLY to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Real trainer: gaussian-splatting package with gsplat backend
# ---------------------------------------------------------------------------

def _try_real_trainer(
    colmap_dir: Path,
    image_dir: Path,
    output_dir: Path,
    max_steps: int,
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Run the ``gaussian-splatting`` trainer with the gsplat CUDA backend.

    Requirements
    ------------
    * torch built with CUDA (torch.cuda.is_available() == True)
    * CUDA Toolkit installed and ``CUDA_HOME`` / ``CUDA_PATH`` env var set
      so that gsplat's JIT compiler can find nvcc.
    * ``gaussian-splatting`` package installed (already in the venv).

    The PLY is saved to ``output_dir/point_cloud/iteration_{max_steps}/point_cloud.ply``
    and the canonical path ``output_dir/point_cloud.ply`` is returned.
    """
    if not torch.cuda.is_available():
        raise ImportError(
            "CUDA is not available. Install PyTorch with a CUDA build "
            "(e.g. torch 2.7.1+cu126) to use the real Gaussian Splatting trainer."
        )

    # Verify nvcc is reachable so we give a clear error before a 30 s JIT attempt.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        raise ImportError(
            "CUDA_HOME (or CUDA_PATH) is not set. Install CUDA Toolkit 12.6 from "
            "https://developer.nvidia.com/cuda-12-6-0-download-archive and set the "
            "environment variable before running the real trainer. "
            "Alternatively, use fallback=True (default) to export from COLMAP sparse points."
        )

    try:
        from gaussian_splatting.train import (  # type: ignore[import-untyped]
            prepare_training,
            training,
        )
    except ImportError as exc:
        raise ImportError(
            f"gaussian-splatting package not available: {exc}. "
            "Run: uv pip install gaussian-splatting"
        ) from exc

    # gaussian-splatting expects the COLMAP workspace root (parent of sparse/0).
    # colmap_dir is sparse/0, so we want the workspace root two levels up.
    # However the package's ColmapCameraDataset looks for `images/` and `sparse/`
    # relative to the source path. We use image_dir's parent workspace.
    workspace_root = colmap_dir.parent.parent  # workspace/sparse/0 → workspace
    source_path = str(workspace_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("gsplat", 0.0, f"Starting Gaussian Splatting training ({max_steps} iters)…")

    # Use the gsplat backend which leverages our installed gsplat rasterizer.
    save_iters = [max_steps // 4, max_steps // 2, max_steps]
    dataset, gaussians, trainer = prepare_training(
        sh_degree=3,
        source=source_path,
        device="cuda",
        mode="densify",
        backend="gsplat",
        load_mask=False,   # no mask images in basic pipeline
        load_depth=False,  # no depth images in basic pipeline
    )

    def _report(step: int, total: int) -> None:
        if progress:
            pct = step / total
            progress("gsplat", pct, f"Training step {step}/{total}")

    # gaussian-splatting's training() handles its own tqdm bar; we hook after.
    training(
        dataset=dataset,
        gaussians=gaussians,
        trainer=trainer,
        destination=str(output_dir),
        iteration=max_steps,
        save_iterations=save_iters,
    )

    if progress:
        progress("gsplat", 0.95, "Training complete, locating output PLY…")

    # Find the final PLY (highest iteration saved).
    ply_candidates = sorted(
        (output_dir / "point_cloud").glob("iteration_*/point_cloud.ply")
    )
    if not ply_candidates:
        raise FileNotFoundError(
            f"Gaussian Splatting training completed but no PLY found in {output_dir}"
        )

    final_ply = ply_candidates[-1]
    # Create a canonical symlink at output_dir/point_cloud.ply for downstream steps.
    canonical = output_dir / "point_cloud.ply"
    if not canonical.exists():
        import shutil
        shutil.copy2(final_ply, canonical)

    if progress:
        progress("gsplat", 1.0, f"Gaussian Splatting PLY saved to {canonical}")

    return canonical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    colmap_dir: Path,
    image_dir: Path,
    output_dir: Path,
    max_steps: int = 30_000,
    resolution: int = -1,
    fallback: bool = True,
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Train a 3D Gaussian Splatting model or export a Gaussian cloud.

    Args:
        colmap_dir: Path to COLMAP sparse/0 directory (must contain points3D.bin).
        image_dir: Path to input images (used by real trainer; fallback ignores it).
        output_dir: Where to save the trained model.
        max_steps: Training iterations (only used by real trainer).
        resolution: Downscale factor (currently unused; real trainer uses full res).
        fallback: If True (default), export a Gaussian Splat PLY directly from
                  COLMAP sparse points instead of running a full training loop.
                  Set to False (or set env var ``FORENSIM_GSPLAT_FALLBACK=0``)
                  after installing the CUDA Toolkit to use the real trainer.
        progress: Optional callback(step_name, percent, message).

    Returns:
        Path to the output .ply file.
    """
    points_path = colmap_dir / "points3D.bin"
    if not points_path.exists():
        raise FileNotFoundError(f"COLMAP points3D.bin not found: {points_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Environment override: FORENSIM_GSPLAT_FALLBACK=0 forces real trainer.
    env_fallback = os.environ.get("FORENSIM_GSPLAT_FALLBACK")
    if env_fallback is not None:
        fallback = env_fallback.strip() not in ("0", "false", "no")

    ply_path = output_dir / "point_cloud.ply"

    if not fallback:
        try:
            return _try_real_trainer(
                colmap_dir=colmap_dir,
                image_dir=image_dir,
                output_dir=output_dir,
                max_steps=max_steps,
                progress=progress,
            )
        except ImportError as exc:
            logger.warning(
                "Real Gaussian Splatting trainer unavailable, switching to fallback: %s", exc
            )
        except Exception:
            logger.exception("Real trainer failed; falling back to point-cloud exporter")

    return _export_splats_from_points(points_path, ply_path, progress=progress)
