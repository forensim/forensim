"""Unit tests for forensim.simulate (scene_builder + physx_runner)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from forensim.simulate.scene_builder import (
    RigidBodySpec,
    SceneBuilderConfig,
    _build_physics_scene_fallback,
    build_physics_scene,
)
from forensim.simulate.physx_runner import (
    SimulationScenario,
    SimulationResult,
    run_monte_carlo,
    run_scenario,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_usd(tmp_path: Path) -> Path:
    """Create a minimal placeholder USD file for testing."""
    usd = tmp_path / "scene.usda"
    usd.write_text("#usda 1.0\ndef Xform \"World\" {}\n")
    return usd


# ── scene_builder tests ───────────────────────────────────────────────────────

class TestSceneBuilderFallback:
    """Tests for _build_physics_scene_fallback (works without usd-core)."""

    def test_copies_usd(self, dummy_usd: Path, tmp_path: Path) -> None:
        out = tmp_path / "physics" / "scene.usda"
        config = SceneBuilderConfig()
        result = _build_physics_scene_fallback(dummy_usd, out, config)
        assert result == out
        assert out.exists()
        assert out.read_text() == dummy_usd.read_text()

    def test_writes_sidecar_json(self, dummy_usd: Path, tmp_path: Path) -> None:
        out = tmp_path / "physics_scene.usda"
        config = SceneBuilderConfig(
            rigid_bodies=[RigidBodySpec(prim_path="/World/box", mass=2.0)],
            gravity=(0.0, -9.81, 0.0),
            ground_plane=True,
        )
        _build_physics_scene_fallback(dummy_usd, out, config)
        sidecar = out.with_suffix(".physics.json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["gravity"] == [0.0, -9.81, 0.0]
        assert data["ground_plane"] is True
        assert len(data["rigid_bodies"]) == 1
        assert data["rigid_bodies"][0]["prim_path"] == "/World/box"
        assert data["rigid_bodies"][0]["mass"] == pytest.approx(2.0)

    def test_creates_parent_dirs(self, dummy_usd: Path, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "scene.usda"
        _build_physics_scene_fallback(dummy_usd, out, SceneBuilderConfig())
        assert out.exists()

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_physics_scene(
                tmp_path / "nonexistent.usda",
                tmp_path / "out.usda",
                SceneBuilderConfig(),
            )

    def test_empty_rigid_bodies(self, dummy_usd: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.usda"
        _build_physics_scene_fallback(dummy_usd, out, SceneBuilderConfig())
        sidecar = out.with_suffix(".physics.json")
        data = json.loads(sidecar.read_text())
        assert data["rigid_bodies"] == []


class TestRigidBodySpec:
    def test_defaults(self) -> None:
        rb = RigidBodySpec(prim_path="/World/obj")
        assert rb.mass == pytest.approx(0.1)
        assert rb.initial_velocity == (0.0, 0.0, 0.0)
        assert rb.collision_approximation == "convexHull"

    def test_custom_values(self) -> None:
        rb = RigidBodySpec(
            prim_path="/World/bullet",
            mass=0.005,
            initial_velocity=(10.0, 0.0, 5.0),
            collision_approximation="boundingCube",
        )
        assert rb.mass == pytest.approx(0.005)
        assert rb.initial_velocity == (10.0, 0.0, 5.0)


# ── physx_runner tests ────────────────────────────────────────────────────────

class TestSimulationScenario:
    def test_defaults(self) -> None:
        s = SimulationScenario(object_name="casing")
        assert s.velocity == (0.0, 0.0, 0.0)
        assert s.angular_velocity == (0.0, 0.0, 0.0)

    def test_custom_velocity(self) -> None:
        s = SimulationScenario(
            object_name="projectile",
            velocity=(5.0, 0.0, -2.0),
        )
        assert s.velocity == (5.0, 0.0, -2.0)


class TestPhysxRunnerDataclasses:
    """Tests for SimulationScenario and SimulationResult dataclasses."""

    def test_simulation_result_structure(self) -> None:
        scenario = SimulationScenario(object_name="bullet", velocity=(1.0, 0.0, 0.0))
        result = SimulationResult(
            scenario=scenario,
            trajectory=[{"body_0": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}],
            final_positions={"body_0": [0.1, 0.0, 0.0]},
        )
        assert result.scenario.object_name == "bullet"
        assert len(result.trajectory) == 1
        assert "body_0" in result.final_positions

    def test_monte_carlo_empty_scenarios(self, dummy_usd: Path) -> None:
        """run_monte_carlo with zero scenarios returns an empty list without calling PhysX."""
        results = run_monte_carlo(dummy_usd, [], n_steps=1)
        assert results == []
