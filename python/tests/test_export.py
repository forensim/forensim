"""Unit tests for the export module (PDF, USD packaging, video)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from forensim.export.report import ReportInputs, generate_pdf_report
from forensim.export.usd import package_usd_scene
from forensim.export.video import generate_flythrough_video


@pytest.fixture
def sample_report_inputs() -> ReportInputs:
    return ReportInputs(
        case_title="Test Case",
        examiner="Unit Tester",
        notes="Smoke test.",
        reconstruction={"usd_path": "D:/scene.usdz", "duration_seconds": 12.3},
        simulation={
            "results": [
                {
                    "scenario": {"object_name": "casing", "velocity": [1.0, 0.0, 0.0]},
                    "trajectory_length": 120,
                    "final_positions": {"casing": [1.0, 2.0, 3.0]},
                }
            ]
        },
        inference={
            "hypotheses": [
                {
                    "rank": 1,
                    "description": "Direct fire",
                    "posterior": 0.75,
                    "bayes_factor": 1.0,
                    "events": ["shot", "fall"],
                }
            ]
        },
    )


def test_generate_pdf_report(sample_report_inputs: ReportInputs) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.pdf"
        generate_pdf_report(sample_report_inputs, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_package_usd_scene() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        usd = Path(tmp) / "scene.usdz"
        usd.write_text("fake usd")
        ply = Path(tmp) / "scene.ply"
        ply.write_text("fake ply")

        zip_path = Path(tmp) / "export.zip"
        package_usd_scene(usd, zip_path)
        assert zip_path.exists()
        assert zip_path.stat().st_size > 0


def test_generate_flythrough_video() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fly.mp4"
        trajectories = [
            {"points": [[0, 0, 0], [1, 1, 1], [2, 0, 1], [3, 1, 0]], "color": "#f59e0b"}
        ]
        generate_flythrough_video(out, trajectories, duration_seconds=1.0, fps=10)
        assert out.exists()
        assert out.stat().st_size > 0
