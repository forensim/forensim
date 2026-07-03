"""Physics simulation API routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ScenarioRequest(BaseModel):
    object_name: str
    velocity: list[float] = [0.0, 0.0, 0.0]
    angular_velocity: list[float] = [0.0, 0.0, 0.0]


class SimulateRequest(BaseModel):
    usd_path: str
    scenarios: list[ScenarioRequest]
    n_steps: int = 300
    dt: float = 1.0 / 60.0


class TrajectoryPoint(BaseModel):
    step: int
    transforms: dict[str, list[float]]


class SimulateResponse(BaseModel):
    status: str
    results: list[dict]


@router.post("/run", response_model=SimulateResponse)
async def run_simulation(req: SimulateRequest) -> SimulateResponse:
    """Run Monte Carlo PhysX simulation across multiple scenarios."""
    usd_path = Path(req.usd_path)
    if not usd_path.exists():
        raise HTTPException(status_code=400, detail=f"USD file not found: {usd_path}")

    try:
        from forensim.simulate.physx_runner import SimulationScenario, run_monte_carlo

        scenarios = [
            SimulationScenario(
                object_name=s.object_name,
                velocity=tuple(s.velocity),          # type: ignore[arg-type]
                angular_velocity=tuple(s.angular_velocity),  # type: ignore[arg-type]
            )
            for s in req.scenarios
        ]

        sim_results = run_monte_carlo(usd_path, scenarios, req.n_steps, req.dt)

        return SimulateResponse(
            status="success",
            results=[
                {
                    "scenario": {
                        "object_name": r.scenario.object_name,
                        "velocity": list(r.scenario.velocity),
                    },
                    "final_positions": r.final_positions,
                    "trajectory_length": len(r.trajectory),
                }
                for r in sim_results
            ],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
