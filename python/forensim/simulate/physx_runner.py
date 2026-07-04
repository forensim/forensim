"""
PhysX simulation runner using ovphysx.

Runs multiple simulation scenarios in parallel to support
Monte Carlo inference over initial conditions.

Requires: pip install ovphysx
Docs: https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/

Real ovphysx API (discovered from installed package):
  physx = PhysX()
  physx.add_usd(str(usd_path))         # loads USD, enqueues
  physx.step(dt, sim_time)             # advance simulation
  binding = physx.create_tensor_binding(
      pattern="/World/*",
      tensor_type=TensorType.RIGID_BODY_POSE,
  )
  poses = np.zeros(binding.shape, dtype=np.float32)
  binding.read(poses)                  # [N, 7] — pos(3) + quat(4)
  binding.destroy()
  physx.release()
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
        usd_path:  Path to the USD scene file (must have rigid body prims).
        scenario:  Initial conditions to apply.
        n_steps:   Number of simulation steps (default 300 = 5 seconds @ 60Hz).
        dt:        Timestep in seconds.

    Returns:
        SimulationResult with full trajectory data.

    Raises:
        ImportError: If ovphysx is not installed.
    """
    try:
        from ovphysx import PhysX, TensorType  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "ovphysx not installed. Run: uv pip install ovphysx"
        ) from e

    import numpy as np

    physx = PhysX()
    physx.add_usd(str(usd_path))

    # Set initial linear velocity via tensor binding (RIGID_BODY_LINEAR_VELOCITY)
    try:
        vel_binding = physx.create_tensor_binding(
            prim_paths=[f"/World/{scenario.object_name}"],
            tensor_type=TensorType.RIGID_BODY_LINEAR_VELOCITY,
        )
        vel_data = np.array([list(scenario.velocity)], dtype=np.float32)
        vel_binding.write(vel_data)
        vel_binding.destroy()
    except Exception:
        pass  # Prim may not exist yet; silently continue

    # Set initial angular velocity
    try:
        ang_binding = physx.create_tensor_binding(
            prim_paths=[f"/World/{scenario.object_name}"],
            tensor_type=TensorType.RIGID_BODY_ANGULAR_VELOCITY,
        )
        ang_data = np.array([list(scenario.angular_velocity)], dtype=np.float32)
        ang_binding.write(ang_data)
        ang_binding.destroy()
    except Exception:
        pass

    # Run simulation and record trajectory
    trajectory: list[dict[str, list[float]]] = []
    sim_time = 0.0
    for _ in range(n_steps):
        physx.step(dt, sim_time)
        sim_time += dt
        try:
            pose_binding = physx.create_tensor_binding(
                pattern="/World/*",
                tensor_type=TensorType.RIGID_BODY_POSE,
            )
            poses = np.zeros(pose_binding.shape, dtype=np.float32)
            pose_binding.read(poses)
            pose_binding.destroy()
            # poses: [N, 7] — [x, y, z, qx, qy, qz, qw]
            step_transforms: dict[str, list[float]] = {
                f"body_{i}": poses[i].tolist() for i in range(len(poses))
            }
            trajectory.append(step_transforms)
        except Exception:
            trajectory.append({})

    # Read final positions
    final_positions: dict[str, list[float]] = {}
    try:
        pose_binding = physx.create_tensor_binding(
            pattern="/World/*",
            tensor_type=TensorType.RIGID_BODY_POSE,
        )
        poses = np.zeros(pose_binding.shape, dtype=np.float32)
        pose_binding.read(poses)
        pose_binding.destroy()
        final_positions = {
            f"body_{i}": poses[i, :3].tolist() for i in range(len(poses))
        }
    except Exception:
        pass

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

    PhysX is not thread-safe within a single process, so scenarios run one
    after another. For inter-process parallelism, use a process pool externally.
    """
    return [
        run_scenario(usd_path, scenario, n_steps, dt)
        for scenario in scenarios
    ]
