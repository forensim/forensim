"""
Blood spatter ballistic simulation using pure-numpy SPH-inspired model.

Models blood droplets as discrete particles flying through air under gravity
and drag, then splatting on a target surface. Uses physically-derived stain
ellipse formulas from bloodstain pattern analysis (BPA) literature.

This is NOT a full Navier-Stokes fluid sim — it is a ballistic droplet model
appropriate for forensic reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class BloodDroplet:
    """A single blood droplet in flight.

    Attributes:
        position: Shape (3,) — x, y, z in metres (y = up).
        velocity: Shape (3,) — m/s.
        radius: Droplet radius in metres (0.5mm – 5mm range).
        mass: Derived from rho_blood * (4/3)*pi*r^3; rho_blood = 1060 kg/m³.
    """

    position: Any  # np.ndarray shape (3,)
    velocity: Any  # np.ndarray shape (3,)
    radius: float
    mass: float


@dataclass
class SpatterConfig:
    """Simulation parameters for blood spatter ballistic model.

    Attributes:
        n_droplets: Number of droplets to simulate.
        dt: Timestep in seconds.
        max_time: Maximum flight time in seconds.
        gravity: Gravitational acceleration m/s² (downward, negative y).
        rho_air: Air density in kg/m³.
        cd_sphere: Drag coefficient for a sphere.
        surface_y: Y-coordinate of the floor surface.
        surface_normal: Floor normal vector (y-up).
        min_radius: Minimum droplet radius in metres (0.5 mm).
        max_radius: Maximum droplet radius in metres (5 mm).
        radius_distribution: Distribution type — "lognormal" or "uniform".
        velocity_spread_angle: Cone half-angle in degrees for velocity spread.
        seed: Random seed for reproducibility.
    """

    n_droplets: int = 200
    dt: float = 1 / 500
    max_time: float = 3.0
    gravity: float = 9.81
    rho_air: float = 1.225
    cd_sphere: float = 0.47
    surface_y: float = 0.0
    surface_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    min_radius: float = 0.0005
    max_radius: float = 0.005
    radius_distribution: str = "lognormal"
    velocity_spread_angle: float = 45.0
    seed: int = 42


@dataclass
class DropletImpact:
    """Result of a single droplet impact.

    Attributes:
        position_2d: Shape (2,) — (x, z) impact coordinates on surface.
        radius: Droplet radius in metres.
        impact_speed: Speed at impact in m/s.
        impact_angle: Angle of incidence in degrees (90 = perpendicular).
        stain_major: Major axis of stain ellipse in metres.
        stain_minor: Minor axis of stain ellipse in metres.
        stain_angle: Orientation of ellipse in degrees from x-axis.
        flight_time: Time of flight in seconds.
    """

    position_2d: Any  # np.ndarray shape (2,)
    radius: float
    impact_speed: float
    impact_angle: float
    stain_major: float
    stain_minor: float
    stain_angle: float
    flight_time: float


@dataclass
class SpatterResult:
    """Output of a blood spatter simulation.

    Attributes:
        config: Simulation configuration used.
        source_position: Shape (3,) — origin point.
        source_velocity: Shape (3,) — mean ejection velocity.
        impacts: List of DropletImpact for all landed droplets.
        n_airborne: Number of droplets that never hit the surface.
        mean_impact_angle: Mean angle of incidence across all impacts.
        pattern_centroid: Shape (2,) — (x, z) centroid of all impacts.
        pattern_spread_radius: RMS distance from centroid in metres.
        estimated_source: Shape (3,) — back-projected area-of-origin.
        direction_vector: Shape (2,) — (x, z) unit vector of main flow direction.
    """

    config: SpatterConfig
    source_position: Any  # np.ndarray shape (3,)
    source_velocity: Any  # np.ndarray shape (3,)
    impacts: list[DropletImpact]
    n_airborne: int
    mean_impact_angle: float
    pattern_centroid: Any  # np.ndarray shape (2,)
    pattern_spread_radius: float
    estimated_source: Any  # np.ndarray shape (3,)
    direction_vector: Any  # np.ndarray shape (2,)


def _compute_droplet_mass(radius: float) -> float:
    """Compute mass from radius using blood density (1060 kg/m³).

    Args:
        radius: Droplet radius in metres.

    Returns:
        Mass in kilograms.
    """
    rho_blood = 1060.0
    return rho_blood * (4.0 / 3.0) * math.pi * radius**3


def _generate_cone_velocities(
    base_velocity: Any,
    half_angle_deg: float,
    n: int,
    rng: Any,
) -> Any:
    """Generate velocity vectors within a cone around the base velocity.

    Args:
        base_velocity: Mean ejection velocity vector shape (3,).
        half_angle_deg: Cone half-angle in degrees.
        n: Number of velocity vectors to generate.
        rng: numpy random Generator instance.

    Returns:
        Array of shape (n, 3) with perturbed velocity vectors.
    """
    import numpy as np

    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    base_speed = float(np.linalg.norm(base_velocity))
    if base_speed < 1e-12:
        # Zero velocity — return zeros
        return np.zeros((n, 3), dtype=np.float64)

    half_angle_rad = math.radians(half_angle_deg)

    # Normalised base direction
    base_dir = base_velocity / base_speed

    # Build orthonormal frame around base_dir
    # Pick an arbitrary vector not parallel to base_dir
    if abs(base_dir[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])

    u = np.cross(base_dir, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(base_dir, u)

    # Random azimuthal angle φ ∈ [0, 2π)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    # Random polar deviation θ from uniform [0, half_angle_rad]
    theta = rng.uniform(0.0, half_angle_rad, size=n)

    # Rodrigues rotation: rotate base_dir by theta around axis in (u, v) plane
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Direction in the cone
    directions = (
        cos_theta[:, None] * base_dir[None, :]
        + sin_theta[:, None] * cos_phi[:, None] * u[None, :]
        + sin_theta[:, None] * sin_phi[:, None] * v[None, :]
    )

    # Speed variation: lognormal(0, 0.3) scale factor
    speed_factors = rng.lognormal(mean=0.0, sigma=0.3, size=n)
    speeds = base_speed * speed_factors

    velocities = directions * speeds[:, None]
    return velocities


def simulate_spatter(
    source_position: Any,
    source_velocity: Any,
    config: SpatterConfig | None = None,
) -> SpatterResult:
    """Run blood spatter ballistic simulation.

    Models blood droplets as discrete particles with aerodynamic drag,
    integrating their trajectories until they impact the floor surface.

    Args:
        source_position: Shape (3,) array — x, y, z origin in metres.
        source_velocity: Shape (3,) array — mean ejection velocity in m/s.
        config: Simulation configuration. Uses defaults if None.

    Returns:
        SpatterResult with all droplet impacts and forensic metrics.
    """
    import numpy as np

    if config is None:
        config = SpatterConfig()

    source_position = np.asarray(source_position, dtype=np.float64)
    source_velocity = np.asarray(source_velocity, dtype=np.float64)

    n = config.n_droplets
    rng = np.random.default_rng(config.seed)

    # Handle zero-droplet edge case
    if n == 0:
        return SpatterResult(
            config=config,
            source_position=source_position,
            source_velocity=source_velocity,
            impacts=[],
            n_airborne=0,
            mean_impact_angle=0.0,
            pattern_centroid=np.zeros(2),
            pattern_spread_radius=0.0,
            estimated_source=source_position.copy(),
            direction_vector=np.array([1.0, 0.0]),
        )

    # ── Step 1: Generate droplet radii ────────────────────────────────────────
    if config.radius_distribution == "lognormal":
        # mu = ln(1.5mm) = ln(0.0015)
        mu = math.log(0.0015)
        sigma = 0.4
        radii = rng.lognormal(mean=mu, sigma=sigma, size=n)
        radii = np.clip(radii, config.min_radius, config.max_radius)
    else:
        radii = rng.uniform(config.min_radius, config.max_radius, size=n)

    # Compute masses: rho_blood * (4/3) * pi * r^3
    rho_blood = 1060.0
    masses = rho_blood * (4.0 / 3.0) * math.pi * radii**3

    # ── Step 2: Generate velocity distribution ────────────────────────────────
    velocities = _generate_cone_velocities(
        source_velocity, config.velocity_spread_angle, n, rng
    )

    # ── Step 3: Ballistic integration (vectorised) ────────────────────────────
    positions = np.tile(source_position, (n, 1))  # (n, 3)
    vels = velocities.copy()  # (n, 3)

    max_steps = int(config.max_time / config.dt)
    active = np.ones(n, dtype=bool)
    impact_times = np.full(n, config.max_time)
    impact_velocities = np.zeros((n, 3), dtype=np.float64)
    impact_positions = np.zeros((n, 3), dtype=np.float64)

    # Pre-compute cross-sectional areas
    areas = math.pi * radii**2  # (n,)

    # Drag coefficient pre-factor: 0.5 * rho_air * cd * A
    drag_coeff = 0.5 * config.rho_air * config.cd_sphere * areas  # (n,)

    gravity_vec = np.array([0.0, -config.gravity, 0.0])

    for step in range(max_steps):
        if not np.any(active):
            break

        # Compute speed magnitude for active droplets
        speed = np.linalg.norm(vels[active], axis=1)  # (n_active,)
        speed_safe = np.maximum(speed, 1e-12)

        # Drag force: F_drag = -drag_coeff * |v| * v
        drag_accel = np.zeros_like(vels[active])
        drag_accel = -(drag_coeff[active, None] / masses[active, None]) * (
            speed_safe[:, None] * vels[active]
        )

        # Net acceleration
        accel = drag_accel + gravity_vec[None, :]

        # Euler integration
        vels[active] += accel * config.dt
        positions[active] += vels[active] * config.dt

        # Check floor crossing
        crossed = active.copy()
        crossed[active] &= positions[active][:, 1] <= config.surface_y

        if np.any(crossed):
            impact_times[crossed] = (step + 1) * config.dt
            impact_velocities[crossed] = vels[crossed]
            impact_positions[crossed] = positions[crossed]
            # Snap y to surface
            impact_positions[crossed, 1] = config.surface_y
            active[crossed] = False

    # Droplets still active after max_time are "airborne" (escaped)
    n_airborne = int(np.sum(active))

    # ── Step 4: Compute stain ellipses ────────────────────────────────────────
    impacts: list[DropletImpact] = []
    landed_mask = ~active

    if np.any(landed_mask):
        landed_indices = np.where(landed_mask)[0]
        imp_vels = impact_velocities[landed_indices]  # (n_landed, 3)
        imp_pos = impact_positions[landed_indices]
        imp_radii = radii[landed_indices]
        imp_times = impact_times[landed_indices]

        # Impact speed
        imp_speed = np.linalg.norm(imp_vels, axis=1)

        # Impact angle: arcsin(|vy| / |v_total|)
        # vy is negative (downward), so we use abs
        vy_abs = np.abs(imp_vels[:, 1])
        imp_speed_safe = np.maximum(imp_speed, 1e-12)
        sin_alpha = np.clip(vy_abs / imp_speed_safe, 0.0, 1.0)
        impact_angles = np.degrees(np.arcsin(sin_alpha))
        # Ensure angles are in (0, 90]
        impact_angles = np.clip(impact_angles, 0.01, 90.0)

        # Stain ellipse dimensions
        stain_minor = 2.0 * imp_radii  # diameter = minor axis (width)
        sin_alpha_rad = np.sin(np.radians(impact_angles))
        sin_alpha_rad = np.maximum(sin_alpha_rad, 1e-6)
        stain_major = stain_minor / sin_alpha_rad
        # Clip major axis to max 20× minor
        stain_major = np.minimum(stain_major, 20.0 * stain_minor)

        # Stain angle: atan2(vx, vz) of impact velocity projected on floor
        stain_angles = np.degrees(np.arctan2(imp_vels[:, 0], imp_vels[:, 2]))

        for i, idx in enumerate(landed_indices):
            impacts.append(
                DropletImpact(
                    position_2d=np.array([imp_pos[i, 0], imp_pos[i, 2]]),
                    radius=float(imp_radii[i]),
                    impact_speed=float(imp_speed[i]),
                    impact_angle=float(impact_angles[i]),
                    stain_major=float(stain_major[i]),
                    stain_minor=float(stain_minor[i]),
                    stain_angle=float(stain_angles[i]),
                    flight_time=float(imp_times[i]),
                )
            )

    # ── Step 5: Area-of-origin estimation ─────────────────────────────────────
    if impacts:
        landed_indices = np.where(landed_mask)[0]
        imp_vels = impact_velocities[landed_indices]
        imp_pos = impact_positions[landed_indices]

        # Back-project along -velocity direction to source_y height
        # parametric: pos + t * (-vel) => y-component: pos_y + t * (-vy) = source_y
        # t = (source_y - pos_y) / (-vy)
        neg_vy = -imp_vels[:, 1]
        # Only back-project for droplets with significant upward back-projection
        valid_bp = np.abs(neg_vy) > 1e-6
        if np.any(valid_bp):
            t_bp = (source_position[1] - imp_pos[valid_bp, 1]) / neg_vy[valid_bp]
            t_bp = np.maximum(t_bp, 0.0)  # Only positive time
            bp_points = imp_pos[valid_bp] + t_bp[:, None] * (-imp_vels[valid_bp])
            estimated_source = np.mean(bp_points, axis=0)
        else:
            estimated_source = source_position.copy()

        # Pattern centroid (x, z)
        pos_2d = np.array([[imp.position_2d[0], imp.position_2d[1]] for imp in impacts])
        pattern_centroid = np.mean(pos_2d, axis=0)

        # Pattern spread: RMS distance from centroid
        dists = np.linalg.norm(pos_2d - pattern_centroid[None, :], axis=1)
        pattern_spread_radius = float(np.sqrt(np.mean(dists**2)))

        # Mean impact angle
        mean_impact_angle = float(np.mean([imp.impact_angle for imp in impacts]))

        # Direction vector: mean of (vx, vz) unit vectors
        floor_vels = imp_vels[:, [0, 2]]  # (n_landed, 2)
        floor_speeds = np.linalg.norm(floor_vels, axis=1, keepdims=True)
        floor_speeds_safe = np.maximum(floor_speeds, 1e-12)
        floor_unit = floor_vels / floor_speeds_safe
        direction_vector = np.mean(floor_unit, axis=0)
        dir_norm = float(np.linalg.norm(direction_vector))
        if dir_norm > 1e-12:
            direction_vector = direction_vector / dir_norm
        else:
            direction_vector = np.array([1.0, 0.0])
    else:
        mean_impact_angle = 0.0
        pattern_centroid = np.zeros(2)
        pattern_spread_radius = 0.0
        estimated_source = source_position.copy()
        direction_vector = np.array([1.0, 0.0])

    return SpatterResult(
        config=config,
        source_position=source_position,
        source_velocity=source_velocity,
        impacts=impacts,
        n_airborne=n_airborne,
        mean_impact_angle=mean_impact_angle,
        pattern_centroid=pattern_centroid,
        pattern_spread_radius=pattern_spread_radius,
        estimated_source=estimated_source,
        direction_vector=direction_vector,
    )


def spatter_to_dict(result: SpatterResult) -> dict[str, Any]:
    """Serialise SpatterResult to a JSON-safe dict for the API.

    Args:
        result: A SpatterResult from simulate_spatter().

    Returns:
        Dictionary with all fields converted to JSON-serialisable types.
    """
    import numpy as np

    def _to_list(arr: Any) -> list[float]:
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return list(arr)

    impacts_list = [
        {
            "position_2d": _to_list(imp.position_2d),
            "radius": imp.radius,
            "impact_speed": imp.impact_speed,
            "impact_angle": imp.impact_angle,
            "stain_major": imp.stain_major,
            "stain_minor": imp.stain_minor,
            "stain_angle": imp.stain_angle,
            "flight_time": imp.flight_time,
        }
        for imp in result.impacts
    ]

    return {
        "config": {
            "n_droplets": result.config.n_droplets,
            "dt": result.config.dt,
            "max_time": result.config.max_time,
            "gravity": result.config.gravity,
            "surface_y": result.config.surface_y,
            "velocity_spread_angle": result.config.velocity_spread_angle,
            "seed": result.config.seed,
        },
        "source_position": _to_list(result.source_position),
        "source_velocity": _to_list(result.source_velocity),
        "impacts": impacts_list,
        "n_impacts": len(result.impacts),
        "n_airborne": result.n_airborne,
        "mean_impact_angle": result.mean_impact_angle,
        "pattern_centroid": _to_list(result.pattern_centroid),
        "pattern_spread_radius": result.pattern_spread_radius,
        "estimated_source": _to_list(result.estimated_source),
        "direction_vector": _to_list(result.direction_vector),
    }
