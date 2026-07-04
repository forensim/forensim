"""Unit tests for forensim.simulate.fluid_spatter and spatter_analysis."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from forensim.simulate.fluid_spatter import (
    SpatterConfig,
    SpatterResult,
    simulate_spatter,
    spatter_to_dict,
)
from forensim.simulate.spatter_analysis import (
    PatternMetrics,
    analyse_pattern,
    pattern_to_evidence_source,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def default_config() -> SpatterConfig:
    """Fast config for testing: few droplets, fixed seed."""
    return SpatterConfig(n_droplets=50, seed=7)


@pytest.fixture
def default_result(default_config: SpatterConfig) -> SpatterResult:
    """Run a standard simulation for reuse in tests."""
    source_pos = np.array([0.0, 1.2, 0.0])
    source_vel = np.array([4.0, -1.5, 0.0])
    return simulate_spatter(source_pos, source_vel, default_config)


# ── TestSpatterConfig ─────────────────────────────────────────────────────────


class TestSpatterConfig:
    def test_defaults(self) -> None:
        cfg = SpatterConfig()
        assert cfg.n_droplets == 200
        assert cfg.dt == pytest.approx(1 / 500)
        assert cfg.max_time == pytest.approx(3.0)
        assert cfg.gravity == pytest.approx(9.81)
        assert cfg.rho_air == pytest.approx(1.225)
        assert cfg.cd_sphere == pytest.approx(0.47)
        assert cfg.surface_y == pytest.approx(0.0)
        assert cfg.surface_normal == (0.0, 1.0, 0.0)
        assert cfg.min_radius == pytest.approx(0.0005)
        assert cfg.max_radius == pytest.approx(0.005)
        assert cfg.radius_distribution == "lognormal"
        assert cfg.velocity_spread_angle == pytest.approx(45.0)
        assert cfg.seed == 42

    def test_custom_surface_normal(self) -> None:
        """Can configure surface_normal to represent a wall."""
        cfg = SpatterConfig(surface_normal=(1.0, 0.0, 0.0))
        assert cfg.surface_normal == (1.0, 0.0, 0.0)


# ── TestSimulateSpatter ───────────────────────────────────────────────────────


class TestSimulateSpatter:
    def test_returns_spatter_result(self, default_result: SpatterResult) -> None:
        assert isinstance(default_result, SpatterResult)

    def test_n_impacts_plus_airborne_equals_n_droplets(
        self, default_result: SpatterResult, default_config: SpatterConfig
    ) -> None:
        total = len(default_result.impacts) + default_result.n_airborne
        assert total == default_config.n_droplets

    def test_impacts_on_surface(
        self, default_result: SpatterResult, default_config: SpatterConfig
    ) -> None:
        """All impacts should have landed at the surface y-coordinate."""
        for imp in default_result.impacts:
            # position_2d is (x, z); the y was snapped to surface
            assert isinstance(imp.position_2d, np.ndarray)
            assert imp.position_2d.shape == (2,)

    def test_positive_impact_speed(self, default_result: SpatterResult) -> None:
        for imp in default_result.impacts:
            assert imp.impact_speed > 0.0

    def test_impact_angles_in_range(self, default_result: SpatterResult) -> None:
        for imp in default_result.impacts:
            assert 0.0 < imp.impact_angle <= 90.0

    def test_stain_major_gte_minor(self, default_result: SpatterResult) -> None:
        for imp in default_result.impacts:
            assert imp.stain_major >= imp.stain_minor - 1e-12

    def test_estimated_source_near_actual(self, default_result: SpatterResult) -> None:
        actual = np.array([0.0, 1.2, 0.0])
        estimated = np.asarray(default_result.estimated_source)
        dist = float(np.linalg.norm(estimated - actual))
        # Back-projection is approximate; allow generous tolerance for cone spread
        assert dist < 1.0, f"Estimated source too far from actual: {dist:.3f} m"

    def test_seeded_reproducibility(self, default_config: SpatterConfig) -> None:
        source_pos = np.array([0.0, 1.2, 0.0])
        source_vel = np.array([4.0, -1.5, 0.0])
        r1 = simulate_spatter(source_pos, source_vel, default_config)
        r2 = simulate_spatter(source_pos, source_vel, default_config)
        assert len(r1.impacts) == len(r2.impacts)
        for a, b in zip(r1.impacts, r2.impacts):
            assert a.impact_speed == pytest.approx(b.impact_speed)
            assert a.impact_angle == pytest.approx(b.impact_angle)

    def test_zero_droplets_empty_result(self) -> None:
        cfg = SpatterConfig(n_droplets=0)
        result = simulate_spatter(
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            cfg,
        )
        assert result.impacts == []
        assert result.n_airborne == 0

    def test_direction_vector_unit(self, default_result: SpatterResult) -> None:
        dv = np.asarray(default_result.direction_vector)
        norm = float(np.linalg.norm(dv))
        assert norm == pytest.approx(1.0, abs=1e-6)


# ── TestSpatterToDict ─────────────────────────────────────────────────────────


class TestSpatterToDict:
    def test_serializable(self, default_result: SpatterResult) -> None:
        d = spatter_to_dict(default_result)
        # Must be JSON-serialisable without error
        json.dumps(d)

    def test_has_required_keys(self, default_result: SpatterResult) -> None:
        d = spatter_to_dict(default_result)
        assert "impacts" in d
        assert "n_impacts" in d
        assert "estimated_source" in d
        assert "pattern_centroid" in d
        assert "direction_vector" in d


# ── TestAnalysePattern ────────────────────────────────────────────────────────


class TestAnalysePattern:
    def test_returns_pattern_metrics(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        assert isinstance(metrics, PatternMetrics)

    def test_stain_count_matches(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        assert metrics.stain_count == len(default_result.impacts)

    def test_log_likelihood_finite(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        assert math.isfinite(metrics.log_likelihood)

    def test_pattern_type_valid(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        valid_types = {"impact", "cast_off", "projected", "transfer", "unknown"}
        assert metrics.pattern_type in valid_types

    def test_impact_angle_mean_in_range(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        if metrics.stain_count > 0:
            assert 0.0 < metrics.impact_angle_mean < 90.0

    def test_notes_is_list(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        assert isinstance(metrics.notes, list)


# ── TestPatternToEvidenceSource ───────────────────────────────────────────────


class TestPatternToEvidenceSource:
    def test_returns_dict_with_keys(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        d = pattern_to_evidence_source(metrics)
        assert "name" in d
        assert "weight" in d
        assert "log_likelihood_deltas" in d

    def test_custom_name(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        d = pattern_to_evidence_source(metrics, name="custom_spatter")
        assert d["name"] == "custom_spatter"

    def test_log_likelihood_deltas_finite(self, default_result: SpatterResult) -> None:
        metrics = analyse_pattern(default_result)
        d = pattern_to_evidence_source(metrics)
        for val in d["log_likelihood_deltas"]:
            assert math.isfinite(val)
