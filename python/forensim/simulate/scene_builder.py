"""
USD scene enrichment for PhysX simulation.

Takes a reconstructed USD scene and annotates it with PhysX properties:
- Rigid body physics on specified objects
- Collision approximation meshes
- PhysX scene-level gravity and solver settings

Requires: usd-core (pxr)
Omniverse PhysX schema docs:
  https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/physx-usd-schema.html
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RigidBodySpec:
    """Specification for making a USD prim a rigid body."""

    prim_path: str
    """USD prim path, e.g. '/World/casing_01'."""

    mass: float = 0.1
    """Mass in kilograms."""

    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Linear velocity in m/s applied at simulation start."""

    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Angular velocity in rad/s applied at simulation start."""

    collision_approximation: str = "convexHull"
    """Collision shape: 'convexHull', 'boundingCube', 'none'."""

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneBuilderConfig:
    """Configuration for the USD scene enrichment pass."""

    rigid_bodies: list[RigidBodySpec] = field(default_factory=list)
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    """Gravity vector in m/s²."""

    ground_plane: bool = True
    """Add an infinite ground plane collider at y=0."""

    solver_iterations: int = 16
    time_steps_per_second: int = 60


def build_physics_scene(
    input_usd: Path,
    output_usd: Path,
    config: SceneBuilderConfig,
) -> Path:
    """
    Enrich a USD scene with PhysX rigid body and scene-level properties.

    Args:
        input_usd:  Path to the source USD scene (from reconstruction).
        output_usd: Path where the enriched USD will be written.
        config:     What rigid bodies and scene settings to apply.

    Returns:
        Path to the written output USD file.

    Raises:
        ImportError: If usd-core (pxr) is not installed.
        FileNotFoundError: If input_usd does not exist.
    """
    if not input_usd.exists():
        raise FileNotFoundError(f"Input USD not found: {input_usd}")

    try:
        from pxr import Gf, UsdPhysics, UsdGeom, Usd, PhysxSchema  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: if full PhysX schema is unavailable, copy + annotate via
        # metadata dicts only so downstream can still read a basic USD.
        return _build_physics_scene_fallback(input_usd, output_usd, config)

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_usd, output_usd)

    stage = Usd.Stage.Open(str(output_usd))

    # ── Scene-level physics ──────────────────────────────────────────────────
    scene_path = "/World/PhysicsScene"
    physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*config.gravity) / 9.81)
    physics_scene.CreateGravityMagnitudeAttr(
        float((sum(v**2 for v in config.gravity) ** 0.5))
    )
    try:
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
        physx_scene.CreateSolverPositionIterationCountAttr(config.solver_iterations)
        physx_scene.CreateTimeStepsPerSecondAttr(config.time_steps_per_second)
    except Exception:
        pass  # PhysxSchema not available — scene still valid for basic physics

    # ── Ground plane ────────────────────────────────────────────────────────
    if config.ground_plane:
        plane_path = "/World/GroundPlane"
        if not stage.GetPrimAtPath(plane_path):
            plane = UsdGeom.Xform.Define(stage, plane_path)
            UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    # ── Rigid bodies ────────────────────────────────────────────────────────
    for rb in config.rigid_bodies:
        prim = stage.GetPrimAtPath(rb.prim_path)
        if not prim.IsValid():
            # Create a placeholder xform if the prim doesn't exist yet
            prim = UsdGeom.Xform.Define(stage, rb.prim_path).GetPrim()

        # Rigid body API
        rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb_api.CreateVelocityAttr(Gf.Vec3f(*rb.initial_velocity))
        rb_api.CreateAngularVelocityAttr(Gf.Vec3f(*rb.initial_angular_velocity))

        # Mass API
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(rb.mass)

        # Collision API
        if rb.collision_approximation != "none":
            col_api = UsdPhysics.CollisionAPI.Apply(prim)
            try:
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.CreateApproximationAttr(rb.collision_approximation)
            except Exception:
                pass  # Not a mesh prim — skip mesh collision settings
            del col_api  # used for side effect

    stage.GetRootLayer().Save()
    return output_usd


def _build_physics_scene_fallback(
    input_usd: Path,
    output_usd: Path,
    config: SceneBuilderConfig,
) -> Path:
    """
    Fallback enrichment when usd-core is not available.

    Copies the USD and writes a JSON sidecar file describing the physics
    properties that would have been applied. Downstream tools can parse this.
    """
    import json

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_usd, output_usd)

    sidecar = output_usd.with_suffix(".physics.json")
    sidecar.write_text(
        json.dumps(
            {
                "gravity": list(config.gravity),
                "ground_plane": config.ground_plane,
                "solver_iterations": config.solver_iterations,
                "time_steps_per_second": config.time_steps_per_second,
                "rigid_bodies": [
                    {
                        "prim_path": rb.prim_path,
                        "mass": rb.mass,
                        "initial_velocity": list(rb.initial_velocity),
                        "initial_angular_velocity": list(rb.initial_angular_velocity),
                        "collision_approximation": rb.collision_approximation,
                    }
                    for rb in config.rigid_bodies
                ],
            },
            indent=2,
        )
    )
    return output_usd
