"""Probabilistic inference API routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

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
        from forensim.infer.sequence import (
            integrate_physx_scores,
            rank_event_sequences,
            rank_event_sequences_hmm,
            compute_posterior_entropy,
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
                        if h.bayes_factor is not None and h.bayes_factor == h.bayes_factor  # NaN check
                        else None
                    ),
                )
                for i, h in enumerate(hypotheses)
            ],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
