"""
Integration test for the reconstruction pipeline.

Uses the small "fountain" multi-view dataset (14 images). The dataset is
downloaded on first run and cached under ``python/tests/fixtures/fountain/``.
This test is marked with ``integration`` and is skipped by default.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pytest

from forensim.reconstruct.pipeline import ReconstructionPipeline


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FOUNTAIN_DIR = FIXTURES_DIR / "fountain"
FOUNTAIN_ZIP = FIXTURES_DIR / "fountain.zip"
FOUNTAIN_URL = "http://vision.maths.lth.se/calledataset/fountain/fountain.zip"


def _ensure_fountain_dataset() -> Path:
    """Download and extract the fountain dataset if it is not cached."""
    if FOUNTAIN_DIR.exists() and any(FOUNTAIN_DIR.iterdir()):
        return FOUNTAIN_DIR

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if not FOUNTAIN_ZIP.exists():
        print(f"Downloading fountain dataset from {FOUNTAIN_URL}...")
        urllib.request.urlretrieve(FOUNTAIN_URL, FOUNTAIN_ZIP)

    with zipfile.ZipFile(FOUNTAIN_ZIP, "r") as zf:
        zf.extractall(FOUNTAIN_DIR)

    return FOUNTAIN_DIR


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("colmap") is None
    and not Path(__import__("os").environ.get("COLMAP_PATH", "")).exists(),
    reason="COLMAP not available",
)
def test_end_to_end_reconstruction(tmp_path: Path) -> None:
    """Run COLMAP -> Gaussian Splat -> USD on the fountain dataset."""
    image_dir = _ensure_fountain_dataset()

    workspace_dir = tmp_path / "workspace"
    pipeline = ReconstructionPipeline(
        matcher="exhaustive",
        gsplat_fallback=True,
    )

    result = pipeline.run(image_dir=image_dir, workspace_dir=workspace_dir)

    assert result.status == "success"
    assert result.ply_path.exists()
    assert result.ply_path.stat().st_size > 0
    assert result.usd_path.exists()
    assert result.usd_path.stat().st_size > 0
    assert (result.workspace_dir / "manifest.json").exists()
    assert result.duration_seconds >= 0
