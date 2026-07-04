"""Physics simulation API routes."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from forensim.simulate.physx_runner import (
    SimulationResult,
    SimulationScenario,
    run_monte_carlo,
)
from forensim.simulate.scene_builder import (
    RigidBodySpec,
    SceneBuilderConfig,
    build_physics_scene,
)

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    object_name: str
    velocity: list[float] = [0.0, 0.0, 0.0]
    angular_velocity: list[float] = [0.0, 0.0, 0.0]


class SimulateRequest(BaseModel):
    usd_path: str
    scenarios: list[ScenarioRequest]
    n_steps: int = 300
    dt: float = 1.0 / 60.0


class SimRunResult(BaseModel):
    scenario: dict[str, object]
    final_positions: dict[str, list[float]]
    trajectory_length: int


class SimulateResponse(BaseModel):
    status: str
    results: list[SimRunResult]
    duration_seconds: float = 0.0


class BuildSceneRequest(BaseModel):
    input_usd: str
    output_usd: str
    rigid_body_paths: list[str] = []
    """USD prim paths to make rigid bodies, e.g. ['/World/casing_01']."""
    masses: list[float] = []
    """Masses for each rigid body (parallel list with rigid_body_paths)."""
    gravity: list[float] = [0.0, -9.81, 0.0]
    ground_plane: bool = True


class BuildSceneResponse(BaseModel):
    status: str
    output_usd: str
    message: str = ""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/build-scene", response_model=BuildSceneResponse)
async def build_scene(req: BuildSceneRequest) -> BuildSceneResponse:
    """Enrich a reconstructed USD with PhysX rigid body properties."""
    input_path = Path(req.input_usd)
    output_path = Path(req.output_usd)

    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"USD not found: {input_path}")

    masses = req.masses if req.masses else [0.1] * len(req.rigid_body_paths)
    if len(masses) < len(req.rigid_body_paths):
        masses = masses + [0.1] * (len(req.rigid_body_paths) - len(masses))

    config = SceneBuilderConfig(
        rigid_bodies=[
            RigidBodySpec(prim_path=path, mass=masses[i])
            for i, path in enumerate(req.rigid_body_paths)
        ],
        gravity=(req.gravity[0], req.gravity[1], req.gravity[2]),
        ground_plane=req.ground_plane,
    )

    try:
        out = await asyncio.get_event_loop().run_in_executor(
            None, build_physics_scene, input_path, output_path, config
        )
        return BuildSceneResponse(
            status="success",
            output_usd=str(out),
            message=f"Physics scene written to {out}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/run", response_model=SimulateResponse)
async def run_simulation(req: SimulateRequest) -> SimulateResponse:
    """Run Monte Carlo PhysX simulation across multiple scenarios."""
    usd_path = Path(req.usd_path)
    if not usd_path.exists():
        raise HTTPException(status_code=400, detail=f"USD file not found: {usd_path}")

    scenarios = [
        SimulationScenario(
            object_name=s.object_name,
            velocity=(s.velocity[0], s.velocity[1], s.velocity[2]),
            angular_velocity=(
                s.angular_velocity[0],
                s.angular_velocity[1],
                s.angular_velocity[2],
            ),
        )
        for s in req.scenarios
    ]

    t0 = time.perf_counter()
    try:
        sim_results: list[SimulationResult] = await asyncio.get_event_loop().run_in_executor(
            None, run_monte_carlo, usd_path, scenarios, req.n_steps, req.dt
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration = time.perf_counter() - t0
    return SimulateResponse(
        status="success",
        duration_seconds=round(duration, 3),
        results=[
            SimRunResult(
                scenario={
                    "object_name": r.scenario.object_name,
                    "velocity": list(r.scenario.velocity),
                },
                final_positions=r.final_positions,
                trajectory_length=len(r.trajectory),
            )
            for r in sim_results
        ],
    )


@router.post("/run-stream")
async def run_simulation_stream(req: SimulateRequest) -> StreamingResponse:
    """
    Run Monte Carlo PhysX simulation with SSE progress streaming.

    Each scenario emits a progress event; when all are done a final
    'done' event is emitted with the full results JSON.
    """
    usd_path = Path(req.usd_path)
    if not usd_path.exists():
        raise HTTPException(status_code=400, detail=f"USD file not found: {usd_path}")

    scenarios = [
        SimulationScenario(
            object_name=s.object_name,
            velocity=(s.velocity[0], s.velocity[1], s.velocity[2]),
            angular_velocity=(
                s.angular_velocity[0],
                s.angular_velocity[1],
                s.angular_velocity[2],
            ),
        )
        for s in req.scenarios
    ]

    async def event_stream() -> object:
        from forensim.simulate.physx_runner import run_scenario

        total = len(scenarios)
        results: list[SimulationResult] = []
        t0 = time.perf_counter()

        for i, scenario in enumerate(scenarios):
            pct = int(i / total * 100)
            yield _sse(
                "progress",
                {"step": f"scenario_{i + 1}", "percent": pct,
                 "message": f"Running scenario {i + 1}/{total}: {scenario.object_name}"},
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, run_scenario, usd_path, scenario, req.n_steps, req.dt
            )
            results.append(result)

        duration = time.perf_counter() - t0
        payload = {
            "status": "success",
            "duration_seconds": round(duration, 3),
            "results": [
                {
                    "scenario": {
                        "object_name": r.scenario.object_name,
                        "velocity": list(r.scenario.velocity),
                    },
                    "final_positions": r.final_positions,
                    "trajectory_length": len(r.trajectory),
                }
                for r in results
            ],
        }
        yield _sse("done", payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Blood spatter simulation ─────────────────────────────────────────────────


class SpatterRequest(BaseModel):
    source_position: list[float] = [0.0, 1.2, 0.0]
    """[x, y, z] in metres."""
    source_velocity: list[float] = [5.0, -2.0, 0.0]
    """[vx, vy, vz] m/s."""
    n_droplets: int = 200
    velocity_spread_angle: float = 45.0
    surface_y: float = 0.0
    max_time: float = 3.0
    seed: int = 42


class SpatterImpactModel(BaseModel):
    position_2d: list[float]
    """[x, z] impact coordinates on surface."""
    radius_mm: float
    """Radius in mm (for display)."""
    impact_speed: float
    impact_angle: float
    stain_major_mm: float
    """Major axis in mm."""
    stain_minor_mm: float
    """Minor axis in mm."""
    stain_angle: float


class SpatterAnalysisModel(BaseModel):
    pattern_type: str
    confidence: float
    source_height: float
    source_distance: float
    mean_stain_area_mm2: float
    stain_count: int
    directionality_score: float
    stringing_factor: float
    impact_angle_mean: float
    log_likelihood: float
    notes: list[str]


class SpatterResponse(BaseModel):
    status: str
    n_impacts: int
    n_airborne: int
    pattern_centroid: list[float]
    """[x, z]."""
    pattern_spread_radius: float
    estimated_source: list[float]
    """[x, y, z]."""
    direction_vector: list[float]
    """[x, z]."""
    impacts: list[SpatterImpactModel]
    analysis: SpatterAnalysisModel
    duration_seconds: float


@router.post("/spatter", response_model=SpatterResponse)
async def run_spatter_simulation(req: SpatterRequest) -> SpatterResponse:
    """Run blood spatter SPH simulation and return droplet impact map + forensic analysis."""
    import numpy as np

    from forensim.simulate.fluid_spatter import SpatterConfig, simulate_spatter
    from forensim.simulate.spatter_analysis import analyse_pattern

    config = SpatterConfig(
        n_droplets=req.n_droplets,
        velocity_spread_angle=req.velocity_spread_angle,
        surface_y=req.surface_y,
        max_time=req.max_time,
        seed=req.seed,
    )

    source_pos = np.array(req.source_position, dtype=np.float64)
    source_vel = np.array(req.source_velocity, dtype=np.float64)

    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, simulate_spatter, source_pos, source_vel, config
    )

    metrics = analyse_pattern(result)
    duration = time.perf_counter() - t0

    # Convert impacts to response models (metres → mm for display)
    impact_models = [
        SpatterImpactModel(
            position_2d=imp.position_2d.tolist()
            if hasattr(imp.position_2d, "tolist")
            else list(imp.position_2d),
            radius_mm=imp.radius * 1000.0,
            impact_speed=imp.impact_speed,
            impact_angle=imp.impact_angle,
            stain_major_mm=imp.stain_major * 1000.0,
            stain_minor_mm=imp.stain_minor * 1000.0,
            stain_angle=imp.stain_angle,
        )
        for imp in result.impacts
    ]

    analysis_model = SpatterAnalysisModel(
        pattern_type=metrics.pattern_type,
        confidence=metrics.confidence,
        source_height=metrics.source_height,
        source_distance=metrics.source_distance,
        mean_stain_area_mm2=metrics.mean_stain_area,
        stain_count=metrics.stain_count,
        directionality_score=metrics.directionality_score,
        stringing_factor=metrics.stringing_factor,
        impact_angle_mean=metrics.impact_angle_mean,
        log_likelihood=metrics.log_likelihood,
        notes=metrics.notes,
    )

    def _to_list(arr: object) -> list[float]:
        if hasattr(arr, "tolist"):
            return arr.tolist()  # type: ignore[union-attr]
        return list(arr)  # type: ignore[arg-type]

    return SpatterResponse(
        status="success",
        n_impacts=len(result.impacts),
        n_airborne=result.n_airborne,
        pattern_centroid=_to_list(result.pattern_centroid),
        pattern_spread_radius=result.pattern_spread_radius,
        estimated_source=_to_list(result.estimated_source),
        direction_vector=_to_list(result.direction_vector),
        impacts=impact_models,
        analysis=analysis_model,
        duration_seconds=round(duration, 3),
    )
