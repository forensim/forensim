"""
3D Gaussian Splatting pipeline.

Provides a training entry point that can either delegate to a real
Gaussian Splatting optimizer (when the heavy dependencies are available)
or fall back to a deterministic point-cloud-to-Gaussian exporter that
creates a valid `.ply` file from COLMAP sparse points. The fallback is
default because it does not require a CUDA build of the full reference
trainer and lets the COLMAP → USD pipeline be verified end-to-end.

The fallback uses ``gsplat.export_splats`` to write a standard PLY file.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


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
        # point3D_id, xyz, rgb, error, track_length
        # Format: QdddBBBdQ + (ii)*track_length
        offset += 8  # point3D_id
        xyz[i] = struct.unpack_from("<3d", data, offset)
        offset += 24
        rgb[i] = struct.unpack_from("<3B", data, offset)
        offset += 3
        offset += 8  # error
        track_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        # track entries: image_id (int32), point2D_idx (int32)
        offset += 8 * track_length

    return xyz, rgb


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


def _try_real_trainer(
    colmap_dir: Path,
    image_dir: Path,
    output_dir: Path,
    max_steps: int,
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Attempt to run a real Gaussian Splatting trainer.

    Currently this is a placeholder that raises ImportError. Once the
    heavy dependencies (a working reference trainer or a custom gsplat
    training loop) are available, this function can be swapped in.
    """
    raise ImportError(
        "Real Gaussian Splatting trainer is not available in this environment. "
        "Use fallback=True or install the full gsplat example dependencies."
    )


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
        image_dir: Path to input images (used by real trainers; fallback ignores it).
        output_dir: Where to save the trained model.
        max_steps: Training iterations (only used by real trainers).
        resolution: Downscale factor (only used by real trainers).
        fallback: If True, export a Gaussian Splat PLY directly from COLMAP
                  sparse points instead of running a full training loop.
        progress: Optional callback(step_name, percent, message).

    Returns:
        Path to the output .ply file.
    """
    points_path = colmap_dir / "points3D.bin"
    if not points_path.exists():
        raise FileNotFoundError(f"COLMAP points3D.bin not found: {points_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
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
            logger.warning("Real trainer unavailable, switching to fallback: %s", exc)

    return _export_splats_from_points(points_path, ply_path, progress=progress)
