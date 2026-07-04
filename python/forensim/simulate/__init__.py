"""forensim.simulate — Physics simulation package."""

from __future__ import annotations

from forensim.simulate.fluid_spatter import (
    BloodDroplet,
    DropletImpact,
    SpatterConfig,
    SpatterResult,
    simulate_spatter,
    spatter_to_dict,
)
from forensim.simulate.physx_runner import (
    SimulationResult,
    SimulationScenario,
    run_monte_carlo,
    run_scenario,
)
from forensim.simulate.scene_builder import (
    RigidBodySpec,
    SceneBuilderConfig,
    build_physics_scene,
)
from forensim.simulate.spatter_analysis import (
    PatternMetrics,
    analyse_pattern,
    pattern_to_evidence_source,
)

__all__ = [
    "SimulationScenario", "SimulationResult", "run_scenario", "run_monte_carlo",
    "RigidBodySpec", "SceneBuilderConfig", "build_physics_scene",
    "BloodDroplet", "SpatterConfig", "DropletImpact", "SpatterResult",
    "simulate_spatter", "spatter_to_dict",
    "PatternMetrics", "analyse_pattern", "pattern_to_evidence_source",
]
