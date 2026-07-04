"""
Forensic bloodstain pattern analysis.

Takes a SpatterResult and extracts forensically-meaningful metrics including
pattern classification, directionality scoring, and log-likelihood computation
for Bayesian hypothesis integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PatternMetrics:
    """Forensic bloodstain pattern analysis metrics.

    Attributes:
        pattern_type: Classification — "cast_off", "impact", "projected",
            "transfer", or "unknown".
        confidence: Confidence score 0-1.
        source_height: Estimated height above floor in metres.
        source_distance: Estimated horizontal distance from pattern centroid.
        mean_stain_area: Average stain area in mm².
        stain_count: Number of stains analysed.
        directionality_score: 0-1 how directional the pattern is
            (1 = very directional).
        satellite_fraction: Fraction of stains < 1mm radius (satellite droplets).
        stringing_factor: Average major/minor axis ratio
            (>3 = directional stringing).
        impact_angle_mean: Mean angle of incidence in degrees.
        impact_angle_std: Standard deviation of impact angles in degrees.
        log_likelihood: Forensic log-likelihood for Bayesian inference.
        notes: Human-readable interpretation notes.
    """

    pattern_type: str
    confidence: float
    source_height: float
    source_distance: float
    mean_stain_area: float
    stain_count: int
    directionality_score: float
    satellite_fraction: float
    stringing_factor: float
    impact_angle_mean: float
    impact_angle_std: float
    log_likelihood: float
    notes: list[str] = field(default_factory=list)


def analyse_pattern(result: Any) -> PatternMetrics:
    """Classify and analyse a blood spatter pattern from a SpatterResult.

    Uses deterministic classification rules for reproducibility. Computes
    forensic metrics including directionality, stringing, satellite fraction,
    and a log-likelihood suitable for Bayesian inference.

    Args:
        result: A SpatterResult from simulate_spatter().

    Returns:
        PatternMetrics with classification and forensic measures.
    """
    import numpy as np

    impacts = result.impacts
    stain_count = len(impacts)

    # Handle empty pattern
    if stain_count == 0:
        return PatternMetrics(
            pattern_type="unknown",
            confidence=0.0,
            source_height=0.0,
            source_distance=0.0,
            mean_stain_area=0.0,
            stain_count=0,
            directionality_score=0.0,
            satellite_fraction=0.0,
            stringing_factor=0.0,
            impact_angle_mean=0.0,
            impact_angle_std=0.0,
            log_likelihood=-10.0,
            notes=["No impacts to analyse."],
        )

    # ── Compute basic metrics ─────────────────────────────────────────────────

    # Stain areas in mm² (ellipse area = pi * a * b, with a/b in metres → convert)
    stain_areas_mm2 = np.array([
        math.pi * (imp.stain_major * 1000.0 / 2.0) * (imp.stain_minor * 1000.0 / 2.0)
        for imp in impacts
    ])
    mean_stain_area = float(np.mean(stain_areas_mm2))

    # Impact angles
    impact_angles = np.array([imp.impact_angle for imp in impacts])
    impact_angle_mean = float(np.mean(impact_angles))
    impact_angle_std = float(np.std(impact_angles)) if stain_count > 1 else 0.0

    # Stringing factor: average major/minor ratio
    axis_ratios = np.array([
        imp.stain_major / max(imp.stain_minor, 1e-9) for imp in impacts
    ])
    stringing_factor = float(np.mean(axis_ratios))

    # Satellite fraction: fraction of stains with radius < 1mm
    satellite_count = sum(1 for imp in impacts if imp.radius < 0.001)
    satellite_fraction = satellite_count / stain_count

    # Directionality score: based on the concentration of stain angles
    # Compute circular variance of stain angles (0 = all same direction, 1 = uniform)
    stain_angles_rad = np.radians([imp.stain_angle for imp in impacts])
    mean_cos = float(np.mean(np.cos(stain_angles_rad)))
    mean_sin = float(np.mean(np.sin(stain_angles_rad)))
    # R (mean resultant length) ranges from 0 (uniform) to 1 (perfectly aligned)
    resultant_length = math.sqrt(mean_cos**2 + mean_sin**2)
    directionality_score = min(resultant_length, 1.0)

    # Source height: from the estimated_source y-coordinate
    source_height = float(result.estimated_source[1]) - result.config.surface_y

    # Source distance: horizontal distance from pattern centroid to estimated source
    est_src_2d = np.array([
        float(result.estimated_source[0]),
        float(result.estimated_source[2]),
    ])
    centroid = np.asarray(result.pattern_centroid, dtype=np.float64)
    source_distance = float(np.linalg.norm(est_src_2d - centroid))

    # ── Pattern classification (deterministic rules) ──────────────────────────
    pattern_type = "unknown"
    confidence = 0.3  # base confidence

    # Rule 1: impact spatter (gunshot / blunt force)
    if directionality_score > 0.6 and stringing_factor > 2.0:
        pattern_type = "impact"
        confidence = 0.5 + 0.3 * directionality_score + 0.1 * min(stringing_factor / 5.0, 1.0)

    # Rule 2: cast-off (weapon swing) — overrides if more specific
    if directionality_score > 0.7 and satellite_fraction < 0.2:
        # Check for arc-shaped pattern: high spread relative to distance
        if result.pattern_spread_radius > 0.1 and source_distance > 0.2:
            pattern_type = "cast_off"
            confidence = 0.6 + 0.3 * directionality_score

    # Rule 3: projected/arterial — overrides previous if matched
    if mean_stain_area > 30.0 and stringing_factor < 1.8:
        pattern_type = "projected"
        confidence = 0.5 + 0.3 * min(mean_stain_area / 100.0, 1.0)

    # Rule 4: transfer/contact
    if stringing_factor < 1.2 and directionality_score < 0.3:
        pattern_type = "transfer"
        confidence = 0.4 + 0.2 * (1.0 - directionality_score)

    confidence = min(confidence, 1.0)

    # ── Log-likelihood for Bayesian integration ───────────────────────────────
    # Base: log(confidence)
    log_likelihood = math.log(max(confidence, 1e-10))

    # Penalise if stain count < 5
    if stain_count < 5:
        log_likelihood -= 0.5 * (5 - stain_count)

    # Bonus if directionality_score > 0.8
    if directionality_score > 0.8:
        log_likelihood += 0.3

    # ── Interpretation notes ──────────────────────────────────────────────────
    notes: list[str] = []
    notes.append(f"Pattern classified as '{pattern_type}' with confidence {confidence:.2f}.")

    if pattern_type == "impact":
        notes.append(
            "High directionality and stringing suggest impact spatter "
            "(e.g., gunshot or blunt force trauma)."
        )
    elif pattern_type == "cast_off":
        notes.append(
            "Directional pattern with low satellite fraction suggests cast-off "
            "(weapon swing arc)."
        )
    elif pattern_type == "projected":
        notes.append(
            "Large stain areas with low stringing suggest projected/arterial spatter."
        )
    elif pattern_type == "transfer":
        notes.append(
            "Low directionality and stringing suggest transfer/contact pattern."
        )
    else:
        notes.append("Pattern does not match standard classification criteria.")

    if satellite_fraction > 0.5:
        notes.append(
            f"High satellite fraction ({satellite_fraction:.0%}) indicates "
            "secondary spatter or high-energy event."
        )

    if impact_angle_mean < 20.0:
        notes.append(
            f"Low mean impact angle ({impact_angle_mean:.1f}°) suggests "
            "oblique/tangential impacts."
        )

    return PatternMetrics(
        pattern_type=pattern_type,
        confidence=confidence,
        source_height=source_height,
        source_distance=source_distance,
        mean_stain_area=mean_stain_area,
        stain_count=stain_count,
        directionality_score=directionality_score,
        satellite_fraction=satellite_fraction,
        stringing_factor=stringing_factor,
        impact_angle_mean=impact_angle_mean,
        impact_angle_std=impact_angle_std,
        log_likelihood=log_likelihood,
        notes=notes,
    )


def pattern_to_evidence_source(
    metrics: PatternMetrics,
    name: str = "blood_spatter",
) -> dict[str, Any]:
    """Convert PatternMetrics into an EvidenceSource-compatible dict.

    For feeding into forensim.infer.sensitivity.compute_sensitivity().
    Returns a dict with keys: name, weight, log_likelihood_deltas.
    log_likelihood_deltas has one entry per hypothesis, all set to
    metrics.log_likelihood (caller should adjust per-hypothesis if they
    have hypothesis-specific info).

    Args:
        metrics: PatternMetrics from analyse_pattern().
        name: Name for the evidence source (default "blood_spatter").

    Returns:
        Dict with keys 'name', 'weight', 'log_likelihood_deltas'.
    """
    return {
        "name": name,
        "weight": metrics.confidence,
        "log_likelihood_deltas": [metrics.log_likelihood],
    }
