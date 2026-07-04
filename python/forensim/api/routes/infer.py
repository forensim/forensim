"""Probabilistic inference API routes."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class AnnotationModel(BaseModel):
    id: str
    image_path: str
    shape: str
    coordinates: list[list[float]]
    tag: str
    description: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = {}


class InferRequest(BaseModel):
    sequences: list[list[str]]
    descriptions: list[str]
    transition_matrix: list[list[float]]
    initial_probs: list[float]
    event_vocab: list[str]
    physx_log_likelihoods: list[float] | None = None
    use_hmm: bool = False
    """If True, use HMM forward log-likelihood instead of Markov chain."""
    emission_matrix: list[list[float]] | None = None
    """Required when use_hmm=True. Shape: [n_states × len(event_vocab)]."""
    annotations: list[AnnotationModel] | None = None
    """Optional evidence ROI annotations that adjust hypothesis likelihoods."""
    annotation_strength: float = 1.0
    """Scaling factor for annotation likelihood effect."""


class HypothesisResult(BaseModel):
    rank: int
    description: str
    log_probability: float
    posterior: float
    events: list[str]
    bayes_factor: float | None = None
    """Bayes factor vs the top-ranked hypothesis. 1.0 for rank 1."""


class InferResponse(BaseModel):
    status: str
    hypotheses: list[HypothesisResult]
    posterior_entropy: float | None = None
    """Shannon entropy (nats) of the posterior distribution."""
    map_description: str | None = None
    """Description of the maximum a-posteriori hypothesis."""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/rank", response_model=InferResponse)
async def rank_hypotheses(req: InferRequest) -> InferResponse:
    """
    Rank candidate event sequences by probabilistic likelihood.

    Optionally integrates PhysX simulation log-likelihoods for
    physics-informed Bayesian ranking.
    """
    if len(req.sequences) != len(req.descriptions):
        raise HTTPException(
            status_code=400,
            detail="sequences and descriptions must have equal length",
        )
    if req.use_hmm and not req.emission_matrix:
        raise HTTPException(
            status_code=400,
            detail="emission_matrix is required when use_hmm=True",
        )

    try:
        from forensim.annotate.manager import Annotation, Shape
        from forensim.infer.sequence import (
            compute_posterior_entropy,
            integrate_annotation_scores,
            integrate_physx_scores,
            rank_event_sequences,
            rank_event_sequences_hmm,
        )

        loop = asyncio.get_event_loop()

        if req.use_hmm and req.emission_matrix:
            hypotheses = await loop.run_in_executor(
                None,
                rank_event_sequences_hmm,
                req.sequences,
                req.descriptions,
                req.transition_matrix,
                req.emission_matrix,
                req.initial_probs,
                req.event_vocab,
            )
        else:
            hypotheses = await loop.run_in_executor(
                None,
                rank_event_sequences,
                req.sequences,
                req.descriptions,
                req.transition_matrix,
                req.initial_probs,
                req.event_vocab,
            )

        if req.physx_log_likelihoods:
            if len(req.physx_log_likelihoods) != len(hypotheses):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"physx_log_likelihoods length ({len(req.physx_log_likelihoods)}) "
                        f"must match sequences length ({len(hypotheses)})"
                    ),
                )
            hypotheses = await loop.run_in_executor(
                None,
                integrate_physx_scores,
                hypotheses,
                req.physx_log_likelihoods,
            )

        if req.annotations:
            annotations = [
                Annotation(
                    id=a.id,
                    image_path=a.image_path,
                    shape=Shape(a.shape),
                    coordinates=a.coordinates,
                    tag=a.tag,
                    description=a.description,
                    confidence=a.confidence,
                    metadata=a.metadata,
                )
                for a in req.annotations
            ]
            hypotheses = await loop.run_in_executor(
                None,
                integrate_annotation_scores,
                hypotheses,
                annotations,
                req.annotation_strength,
            )

        posteriors = [h.posterior for h in hypotheses]
        entropy = compute_posterior_entropy(posteriors)
        map_hyp = hypotheses[0] if hypotheses else None

        return InferResponse(
            status="success",
            posterior_entropy=round(entropy, 6),
            map_description=map_hyp.description if map_hyp else None,
            hypotheses=[
                HypothesisResult(
                    rank=i + 1,
                    description=h.description,
                    log_probability=round(h.log_probability, 6),
                    posterior=round(h.posterior, 6),
                    events=h.events,
                    bayes_factor=(
                        round(h.bayes_factor, 6)
                        if h.bayes_factor is not None and h.bayes_factor == h.bayes_factor
                        else None
                    ),  # NaN check
                )
                for i, h in enumerate(hypotheses)
            ],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Sensitivity analysis models ───────────────────────────────────────────────


class EvidenceSourceModel(BaseModel):
    name: str
    log_likelihood_delta: list[float]
    weight: float = 1.0


class SensitivityRequest(BaseModel):
    hypotheses: list[HypothesisResult]
    evidence_sources: list[EvidenceSourceModel]


class SensitivityResultModel(BaseModel):
    evidence_name: str
    baseline_top_posterior: float
    loo_top_posterior: float
    impact: float
    impact_pct: float
    rank_change: int


class SensitivityResponse(BaseModel):
    status: str
    results: list[SensitivityResultModel]
    baseline_top_posterior: float
    top_hypothesis: str | None = None


# ── Sensitivity route ─────────────────────────────────────────────────────────


@router.post("/sensitivity", response_model=SensitivityResponse)
async def compute_sensitivity_route(req: SensitivityRequest) -> SensitivityResponse:
    """
    Compute leave-one-out sensitivity of each evidence source on the top
    hypothesis posterior.
    """
    try:
        from forensim.infer.sensitivity import EvidenceSource, compute_sensitivity
        from forensim.infer.sequence import ScoredHypothesis

        loop = asyncio.get_event_loop()

        hypotheses: list[ScoredHypothesis] = []
        for i, h in enumerate(req.hypotheses):
            hypotheses.append(
                ScoredHypothesis(
                    index=i,
                    description=h.description,
                    log_probability=h.log_probability,
                    posterior=h.posterior,
                    events=h.events,
                    bayes_factor=h.bayes_factor if h.bayes_factor is not None else float("nan"),
                )
            )

        evidence_sources = [
            EvidenceSource(
                name=e.name,
                log_likelihood_delta=e.log_likelihood_delta,
                weight=e.weight,
            )
            for e in req.evidence_sources
        ]

        results = await loop.run_in_executor(
            None,
            compute_sensitivity,
            hypotheses,
            evidence_sources,
        )

        top_hypothesis = next(
            (h.description for h in req.hypotheses if h.rank == 1),
            None,
        )
        baseline_top_posterior = (
            results[0].baseline_top_posterior if results else 0.0
        )

        return SensitivityResponse(
            status="success",
            baseline_top_posterior=baseline_top_posterior,
            top_hypothesis=top_hypothesis,
            results=[
                SensitivityResultModel(
                    evidence_name=r.evidence_name,
                    baseline_top_posterior=round(r.baseline_top_posterior, 6),
                    loo_top_posterior=round(r.loo_top_posterior, 6),
                    impact=round(r.impact, 6),
                    impact_pct=round(r.impact_pct, 6),
                    rank_change=r.rank_change,
                )
                for r in results
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
