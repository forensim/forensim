"""Tests for forensim.infer.sensitivity."""

from __future__ import annotations

import math

import pytest
from forensim.infer.sensitivity import EvidenceSource, SensitivityResult, compute_sensitivity
from forensim.infer.sequence import ScoredHypothesis


def _make_hyps(log_probs: list[float]) -> list[ScoredHypothesis]:
    hyps = []
    for i, lp in enumerate(log_probs):
        hyps.append(
            ScoredHypothesis(
                index=i,
                description=f"H{i}",
                log_probability=lp,
                posterior=0.0,
                events=[f"event_{i}"],
            )
        )
    # normalise posteriors
    import numpy as np

    arr = np.array(log_probs)
    arr -= arr.max()
    p = np.exp(arr)
    p /= p.sum()
    for h, pi in zip(hyps, p):
        h.posterior = float(pi)
    return hyps


def test_compute_sensitivity_empty_evidence() -> None:
    hyps = _make_hyps([-1.0, -3.0])
    results = compute_sensitivity(hyps, [])
    assert results == []


def test_compute_sensitivity_single_evidence() -> None:
    hyps = _make_hyps([-1.0, -3.0])
    evidence = [EvidenceSource(name="roi_1", log_likelihood_delta=[1.0, 0.0])]
    results = compute_sensitivity(hyps, evidence)
    assert len(results) == 1
    r = results[0]
    baseline_top = math.exp(-1.0) / (math.exp(-1.0) + math.exp(-3.0))
    loo_top = math.exp(-2.0) / (math.exp(-2.0) + math.exp(-3.0))
    assert math.isclose(r.baseline_top_posterior, baseline_top, rel_tol=1e-6)
    assert math.isclose(r.loo_top_posterior, loo_top, rel_tol=1e-6)
    assert math.isclose(r.impact, baseline_top - loo_top, rel_tol=1e-6)
    assert r.impact > 0
    assert r.rank_change == 0


def test_compute_sensitivity_neutral_evidence() -> None:
    hyps = _make_hyps([-1.0, -3.0])
    evidence = [EvidenceSource(name="neutral", log_likelihood_delta=[0.0, 0.0])]
    results = compute_sensitivity(hyps, evidence)
    assert len(results) == 1
    r = results[0]
    assert math.isclose(r.impact, 0.0, abs_tol=1e-6)
    assert math.isclose(r.impact_pct, 0.0, abs_tol=1e-6)
    assert r.rank_change == 0


def test_sensitivity_sorted_by_abs_impact() -> None:
    """Results must be sorted by abs(impact) descending."""
    hyps = _make_hyps([0.0, -2.0])
    # Three sources with clearly different magnitudes:
    #   "big"    adds +10 log-units to top hyp   → large positive impact
    #   "medium" adds +1  log-unit  to top hyp   → moderate positive impact
    #   "small"  adds 0   log-units to both hyps  → zero impact (neutral)
    evidence_sources = [
        EvidenceSource(name="big", log_likelihood_delta=[10.0, 0.0]),
        EvidenceSource(name="medium", log_likelihood_delta=[1.0, 0.0]),
        EvidenceSource(name="small", log_likelihood_delta=[0.0, 0.0]),
    ]
    results = compute_sensitivity(hyps, evidence_sources)
    assert len(results) == 3
    # Sorted descending by absolute impact
    impacts = [abs(r.impact) for r in results]
    assert impacts == sorted(impacts, reverse=True)
    # "big" must dominate
    assert results[0].evidence_name == "big"
    # "small" (neutral) must have lowest absolute impact
    assert results[-1].evidence_name == "small"


def test_sensitivity_result_fields() -> None:
    hyps = _make_hyps([-1.0, -3.0, -5.0])
    evidence = [
        EvidenceSource(name="field_test", log_likelihood_delta=[0.5, 0.0, 0.0])
    ]
    results = compute_sensitivity(hyps, evidence)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, SensitivityResult)
    assert isinstance(r.evidence_name, str)
    assert isinstance(r.baseline_top_posterior, float)
    assert isinstance(r.loo_top_posterior, float)
    assert isinstance(r.impact, float)
    assert isinstance(r.impact_pct, float)
    assert isinstance(r.rank_change, int)
    assert 0.0 <= r.baseline_top_posterior <= 1.0
    assert 0.0 <= r.loo_top_posterior <= 1.0
    assert r.impact_pct == pytest.approx(r.impact / r.baseline_top_posterior * 100, rel=1e-6)
