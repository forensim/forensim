"""
PhysX simulation runner using ovphysx.

Runs multiple simulation scenarios in parallel to support
Monte Carlo inference over initial conditions.

Requires: pip install ovphysx
Docs: https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SimulationScenario:
    """Initial conditions for a single simulation run."""
    object_name: str
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Output of a single simulation run."""
    scenario: SimulationScenario
    # List of per-timestep transforms: [{object_name: [x,y,z,qw,qx,qy,qz]}]
    trajectory: list[dict[str, list[float]]]
    final_positions: dict[str, list[float]]


def run_scenario(
    usd_path: Path,
    scenario: SimulationScenario,
    n_steps: int = 300,
    dt: float = 1.0 / 60.0,
) -> SimulationResult:
    """
    Run a single PhysX simulation scenario.

    Args:
        usd_path:  Path to the USD scene file.
        scenario:  Initial conditions to apply.
        n_steps:   Number of simulation steps (default 300 = 5 seconds @ 60Hz).
        dt:        Timestep in seconds.

    Returns:
        SimulationResult with full trajectory data.
    """
    try:
        from ovphysx import PhysX  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "ovphysx not installed. Run: uv pip install ovphysx"
        ) from e

    import numpy as np

    physx = PhysX()
    physx.add_usd(str(usd_path))

    # Apply initial conditions
    vel = np.array(scenario.velocity, dtype=np.float32)
    physx.set_initial_state(scenario.object_name, velocity=vel)

    trajectory: list[dict[str, list[float]]] = []
    for _ in range(n_steps):
        physx.step(dt, 0.0)
        transforms = physx.get_transforms()
        trajectory.append(
            {name: list(t) for name, t in transforms.items()}
        )

    final_positions = {
        name: list(t[:3]) for name, t in physx.get_transforms().items()
    }
    physx.release()

    return SimulationResult(
        scenario=scenario,
        trajectory=trajectory,
        final_positions=final_positions,
    )


def run_monte_carlo(
    usd_path: Path,
    scenarios: list[SimulationScenario],
    n_steps: int = 300,
    dt: float = 1.0 / 60.0,
) -> list[SimulationResult]:
    """
    Run multiple scenarios sequentially and return all results.

    For true parallelism, call run_scenario from a process pool.
    PhysX is not thread-safe within a single process.
    """
    return [
        run_scenario(usd_path, scenario, n_steps, dt)
        for scenario in scenarios
    ]
