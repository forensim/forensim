"""Unit tests for the reconstruction pipeline."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from forensim.reconstruct import colmap, gsplat, pipeline, usd_export


class TestColmapExe:
    def test_colmap_path_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COLMAP_PATH", r"C:\fake\colmap.exe")
        assert colmap._colmap_exe() == r"C:\fake\colmap.exe"

    def test_colmap_not_found_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "")
        monkeypatch.delenv("COLMAP_PATH", raising=False)
        with pytest.raises(RuntimeError, match="COLMAP not found"):
            colmap._colmap_exe()


class TestReadColmapPoints:
    def test_read_binary_points(self, tmp_path: Path) -> None:
        points_path = tmp_path / "points3D.bin"
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

        with open(points_path, "wb") as f:
            f.write(struct.pack("<Q", len(xyz)))
            for i in range(len(xyz)):
                f.write(struct.pack("<Q", i + 1))  # point3D_id
                f.write(struct.pack("<3d", *xyz[i]))
                f.write(struct.pack("<3B", *rgb[i]))
                f.write(struct.pack("<d", 0.5))  # error
                f.write(struct.pack("<Q", 0))  # track_length

        read_xyz, read_rgb = gsplat._read_colmap_points3d_bin(points_path)
        np.testing.assert_array_equal(read_xyz, xyz)
        np.testing.assert_array_equal(read_rgb, rgb)


class TestGsplatFallback:
    def test_export_splats_from_points(self, tmp_path: Path) -> None:
        points_path = tmp_path / "points3D.bin"
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        rgb = np.full((3, 3), 128, dtype=np.uint8)

        with open(points_path, "wb") as f:
            f.write(struct.pack("<Q", len(xyz)))
            for i in range(len(xyz)):
                f.write(struct.pack("<Q", i + 1))
                f.write(struct.pack("<3d", *xyz[i]))
                f.write(struct.pack("<3B", *rgb[i]))
                f.write(struct.pack("<d", 0.5))
                f.write(struct.pack("<Q", 0))

        output_path = tmp_path / "out.ply"
        result = gsplat._export_splats_from_points(points_path, output_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_empty_point_cloud_raises(self, tmp_path: Path) -> None:
        points_path = tmp_path / "points3D.bin"
        with open(points_path, "wb") as f:
            f.write(struct.pack("<Q", 0))

        with pytest.raises(RuntimeError, match="empty"):
            gsplat._export_splats_from_points(points_path, tmp_path / "out.ply")


class TestUsdExport:
    def test_ply_to_usd_missing_input(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            usd_export.ply_to_usd(tmp_path / "missing.ply", tmp_path / "out.usdz")


class TestPipeline:
    def test_run_success(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        # Create a dummy image so input validation passes.
        (image_dir / "img.jpg").write_bytes(b"fake")

        workspace_dir = tmp_path / "workspace"

        with mock.patch("forensim.reconstruct.colmap.full_pipeline") as mock_colmap:
            mock_colmap.return_value = workspace_dir / "sparse"
            (workspace_dir / "sparse" / "0").mkdir(parents=True)
            (workspace_dir / "sparse" / "0" / "points3D.bin").write_bytes(b"")

            with mock.patch("forensim.reconstruct.gsplat.train") as mock_train:
                ply_path = workspace_dir / "gsplat" / "point_cloud.ply"
                ply_path.parent.mkdir(parents=True)
                ply_path.write_bytes(b"fake")
                mock_train.return_value = ply_path

                with mock.patch("forensim.reconstruct.usd_export.ply_to_usd") as mock_usd:
                    usd_path = workspace_dir / "scene.usdz"
                    mock_usd.return_value = usd_path

                    p = pipeline.ReconstructionPipeline()
                    result = p.run(image_dir=image_dir, workspace_dir=workspace_dir)

        assert result.status == "success"
        assert result.usd_path == workspace_dir / "scene.usdz"
        assert (workspace_dir / "manifest.json").exists()

    def test_run_invalid_image_dir(self, tmp_path: Path) -> None:
        p = pipeline.ReconstructionPipeline()
        with pytest.raises(FileNotFoundError):
            p.run(image_dir=tmp_path / "does_not_exist", workspace_dir=tmp_path / "workspace")

    def test_run_colmap_failure_writes_manifest(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        (image_dir / "img.jpg").write_bytes(b"fake")
        workspace_dir = tmp_path / "workspace"

        with mock.patch(
            "forensim.reconstruct.colmap.full_pipeline",
            side_effect=RuntimeError("COLMAP crashed"),
        ):
            p = pipeline.ReconstructionPipeline()
            with pytest.raises(RuntimeError, match="COLMAP crashed"):
                p.run(image_dir=image_dir, workspace_dir=workspace_dir)

        manifest_path = workspace_dir / "manifest.json"
        assert manifest_path.exists()
        assert "failed" in manifest_path.read_text(encoding="utf-8")
