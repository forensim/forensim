"""
COLMAP wrapper — Structure-from-Motion (SfM) pipeline.

Runs COLMAP to extract camera poses and a sparse point cloud from a
directory of images. COLMAP must be installed separately and made
available via PATH or the ``COLMAP_PATH`` environment variable.

Typical output layout:
    workspace_dir/
        database.db
        sparse/
            0/
                cameras.bin
                images.bin
                points3D.bin
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


def _colmap_exe() -> str:
    """Return the COLMAP executable path, checking COLMAP_PATH env var first."""
    env_path = os.environ.get("COLMAP_PATH")
    if env_path:
        return env_path
    exe = shutil.which("colmap")
    if not exe:
        raise RuntimeError(
            "COLMAP not found. Install from https://colmap.github.io/ "
            "and add its bin folder to PATH, or set COLMAP_PATH "
            "to the colmap executable."
        )
    return exe


def _run_colmap(
    args: list[str],
    step_name: str,
    progress: ProgressCallback | None = None,
    message: str = "",
) -> None:
    """Run a COLMAP subcommand and stream progress."""
    exe = _colmap_exe()
    cmd = [exe] + args
    logger.info("Running COLMAP: %s", " ".join(cmd))

    if progress:
        progress(step_name, 0.0, f"Starting {message}...")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None
    last_line = ""
    for line in process.stdout:
        line = line.rstrip()
        if line:
            last_line = line
            logger.debug("COLMAP %s: %s", step_name, line)
            if progress:
                progress(step_name, -1.0, line)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"COLMAP {step_name} failed (exit {process.returncode}): {last_line}"
        )

    if progress:
        progress(step_name, 1.0, f"Finished {message}.")


def run_feature_extraction(
    image_dir: Path,
    database_path: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """Run COLMAP feature extraction."""
    _run_colmap(
        [
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.single_camera", "1",
        ],
        step_name="feature_extraction",
        progress=progress,
        message="feature extraction",
    )


def run_exhaustive_matcher(
    database_path: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """Run COLMAP exhaustive feature matching."""
    _run_colmap(
        [
            "exhaustive_matcher",
            "--database_path", str(database_path),
        ],
        step_name="exhaustive_matcher",
        progress=progress,
        message="exhaustive matching",
    )


def run_sequential_matcher(
    database_path: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """Run COLMAP sequential feature matching (for video frames)."""
    _run_colmap(
        [
            "sequential_matcher",
            "--database_path", str(database_path),
        ],
        step_name="sequential_matcher",
        progress=progress,
        message="sequential matching",
    )


def run_mapper(
    database_path: Path,
    image_dir: Path,
    output_dir: Path,
    progress: ProgressCallback | None = None,
) -> None:
    """Run COLMAP incremental mapper (sparse reconstruction)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_colmap(
        [
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(output_dir),
        ],
        step_name="mapper",
        progress=progress,
        message="sparse mapping",
    )


def full_pipeline(
    image_dir: Path,
    workspace_dir: Path,
    matcher: str = "exhaustive",
    progress: ProgressCallback | None = None,
) -> Path:
    """
    Run the full COLMAP SfM pipeline.

    Args:
        image_dir: Directory containing input images.
        workspace_dir: Output workspace directory.
        matcher: 'exhaustive' (unordered images) or 'sequential' (video).
        progress: Optional callback(step_name, percent, message).
                  percent < 0 means "indeterminate / log line".

    Returns:
        Path to the sparse reconstruction directory (workspace_dir/sparse).
    """
    _ = _colmap_exe()  # fail fast if COLMAP is missing

    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"

    run_feature_extraction(image_dir, database_path, progress=progress)

    if matcher == "sequential":
        run_sequential_matcher(database_path, progress=progress)
    else:
        run_exhaustive_matcher(database_path, progress=progress)

    run_mapper(database_path, image_dir, sparse_dir, progress=progress)
    return sparse_dir
