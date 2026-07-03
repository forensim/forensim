"""
COLMAP wrapper — Structure-from-Motion (SfM) pipeline.

Runs COLMAP to extract camera poses and a sparse point cloud from a
directory of images. COLMAP must be installed and available on PATH.

Typical output layout:
    output_dir/
        sparse/0/
            cameras.bin
            images.bin
            points3D.bin
        database.db
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_feature_extraction(image_dir: Path, database_path: Path) -> None:
    """Run COLMAP feature extraction."""
    _run_colmap([
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
    ])


def run_exhaustive_matcher(database_path: Path) -> None:
    """Run COLMAP exhaustive feature matching."""
    _run_colmap([
        "exhaustive_matcher",
        "--database_path", str(database_path),
    ])


def run_sequential_matcher(database_path: Path) -> None:
    """Run COLMAP sequential feature matching (for video frames)."""
    _run_colmap([
        "sequential_matcher",
        "--database_path", str(database_path),
    ])


def run_mapper(database_path: Path, image_dir: Path, output_dir: Path) -> None:
    """Run COLMAP incremental mapper (sparse reconstruction)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_colmap([
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_dir),
        "--output_path", str(output_dir),
    ])


def full_pipeline(
    image_dir: Path,
    workspace_dir: Path,
    matcher: str = "exhaustive",
) -> Path:
    """
    Run the full COLMAP SfM pipeline.

    Args:
        image_dir:     Directory containing input images.
        workspace_dir: Output workspace directory.
        matcher:       'exhaustive' (unordered images) or 'sequential' (video).

    Returns:
        Path to the sparse reconstruction directory (workspace_dir/sparse).
    """
    if not shutil.which("colmap"):
        raise RuntimeError(
            "COLMAP not found on PATH. Install from https://colmap.github.io/"
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"

    run_feature_extraction(image_dir, database_path)

    if matcher == "sequential":
        run_sequential_matcher(database_path)
    else:
        run_exhaustive_matcher(database_path)

    run_mapper(database_path, image_dir, sparse_dir)
    return sparse_dir


def _run_colmap(args: list[str]) -> None:
    cmd = ["colmap"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP failed (exit {result.returncode}):\n{result.stderr}"
        )
