"""
Sensitivity analysis for forensic hypothesis ranking.

Computes the marginal influence of each evidence source on the posterior
probability of the top-ranked hypothesis by performing leave-one-out (LOO)
re-ranking and measuring the change in top-hypothesis posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from forensim.infer.sequence import ScoredHypothesis


@dataclass
class EvidenceSource:
    """A single source of evidence and its per-hypothesis log-likelihood contribution."""

    name: str
    """Human-readable label, e.g. 'blood_spatter_roi'."""

    log_likelihood_delta: list[float]
    """Per-hypothesis additive log-likelihood contribution."""

    weight: float = 1.0
    """Current weighting of this evidence source."""


@dataclass
class SensitivityResult:
    """Sensitivity of the top-ranked hypothesis to removing one evidence source."""

    evidence_name: str
    baseline_top_posterior: float
    """P(top_hyp | all evidence)."""

    loo_top_posterior: float
    """P(top_hyp | all evidence except this one)."""

    impact: float
    """baseline_top_posterior - loo_top_posterior (positive = increases confidence)."""

    impact_pct: float
    """impact / baseline_top_posterior * 100."""

    rank_change: int
    """How many positions the top hypothesis drops when this evidence is removed."""


def _softmax(log_probs: list[float]) -> np.ndarray:
    """Numerically stable softmax in log space."""
    arr = np.array(log_probs, dtype=float)
    finite = arr.copy()
    finite[~np.isfinite(finite)] = -1e300
    finite -= finite.max()
    weights = cast(np.ndarray, np.exp(finite))
    total = weights.sum()
    if total <= 0:
        return cast(np.ndarray, np.ones(len(finite), dtype=float) / len(finite))
    return cast(np.ndarray, weights / total)


def compute_sensitivity(
    hypotheses: list[ScoredHypothesis],
    evidence_sources: list[EvidenceSource],
) -> list[SensitivityResult]:
    """
    Compute leave-one-out sensitivity for each evidence source.

    Args:
        hypotheses: Already-ranked hypotheses with log_probabilities set.
        evidence_sources: Evidence sources whose contribution should be evaluated.

    Returns:
        List of SensitivityResult sorted by absolute impact (largest first).
    """
    if not evidence_sources:
        return []

    baseline_log_probs = [h.log_probability for h in hypotheses]
    baseline_posteriors = _softmax(baseline_log_probs)
    baseline_top_idx = int(np.argmax(baseline_posteriors))
    baseline_top_posterior = float(baseline_posteriors[baseline_top_idx])

    results: list[SensitivityResult] = []
    for evidence in evidence_sources:
        new_log_probs = [
            h.log_probability - evidence.weight * evidence.log_likelihood_delta[i]
            for i, h in enumerate(hypotheses)
        ]
        loo_posteriors = _softmax(new_log_probs)
        loo_top_posterior = float(loo_posteriors[baseline_top_idx])

        # Rank of the baseline top hypothesis in the LOO ordering (descending posterior).
        loo_ranks = np.argsort(-loo_posteriors)
        loo_rank = int(np.where(loo_ranks == baseline_top_idx)[0][0])
        rank_change = loo_rank

        impact = baseline_top_posterior - loo_top_posterior
        if baseline_top_posterior > 0:
            impact_pct = impact / baseline_top_posterior * 100.0
        else:
            impact_pct = 0.0

        results.append(
            SensitivityResult(
                evidence_name=evidence.name,
                baseline_top_posterior=baseline_top_posterior,
                loo_top_posterior=loo_top_posterior,
                impact=impact,
                impact_pct=impact_pct,
                rank_change=rank_change,
            )
        )

    results.sort(key=lambda r: abs(r.impact), reverse=True)
    return results


def compute_annotation_sensitivity(
    hypotheses: list[ScoredHypothesis],
    annotations: list[Any],
    strength: float = 1.0,
) -> list[SensitivityResult]:
    """
    Convenience wrapper: group annotations by tag and compute sensitivity.

    Args:
        hypotheses: Already-ranked hypotheses.
        annotations: Evidence annotations (forensim.annotate.manager.Annotation).
        strength: Scaling factor passed to the annotation likelihood model.

    Returns:
        SensitivityResult list per unique annotation tag.
    """
    from forensim.annotate.weights import apply_annotation_likelihoods

    # Group annotations by tag.
    tag_groups: dict[str, list[Any]] = {}
    for annotation in annotations:
        tag = getattr(annotation, "tag", "unknown")
        tag_groups.setdefault(tag, []).append(annotation)

    evidence_sources: list[EvidenceSource] = []
    for tag, tag_annotations in tag_groups.items():
        deltas = apply_annotation_likelihoods(hypotheses, tag_annotations, strength)
        evidence_sources.append(
            EvidenceSource(name=tag, log_likelihood_delta=deltas, weight=1.0)
        )

    return compute_sensitivity(hypotheses, evidence_sources)
