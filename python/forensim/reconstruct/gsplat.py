"""
3D Gaussian Splatting pipeline using the gsplat library.

Trains a Gaussian Splat model from COLMAP output and exports a .ply file.
Requires: pip install gsplat torch
"""

from __future__ import annotations

from pathlib import Path


def train(
    colmap_dir: Path,
    image_dir: Path,
    output_dir: Path,
    max_steps: int = 30_000,
    resolution: int = -1,
) -> Path:
    """
    Train a 3D Gaussian Splatting model.

    Args:
        colmap_dir:   Path to COLMAP sparse/0 directory.
        image_dir:    Path to input images.
        output_dir:   Where to save the trained model.
        max_steps:    Training iterations (default 30k matches gsplat defaults).
        resolution:   Downscale factor (-1 = auto).

    Returns:
        Path to the output .ply file.
    """
    try:
        from gsplat.strategy import DefaultStrategy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "gsplat is not installed. Run: uv pip install gsplat"
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)

    # gsplat's simple_trainer CLI is the canonical entrypoint.
    # We delegate to it as a subprocess so we don't have to replicate
    # the full training loop here — this keeps us compatible with
    # gsplat API changes.
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "gsplat.simple_trainer",
        "default",
        "--data_dir", str(colmap_dir.parent),  # expects colmap/ subdir
        "--data_factor", str(max(1, resolution)),
        "--result_dir", str(output_dir),
        "--max_steps", str(max_steps),
        "--disable_viewer",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("gsplat training failed — see output above")

    # gsplat writes the final model as point_cloud.ply
    ply_path = output_dir / "ply" / "point_cloud.ply"
    if not ply_path.exists():
        # Fallback: find any .ply in output
        candidates = list(output_dir.rglob("*.ply"))
        if not candidates:
            raise FileNotFoundError(f"No .ply output found under {output_dir}")
        ply_path = candidates[0]

    return ply_path
