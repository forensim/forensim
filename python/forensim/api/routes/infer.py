"""Probabilistic inference API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class InferRequest(BaseModel):
    sequences: list[list[str]]
    descriptions: list[str]
    transition_matrix: list[list[float]]
    initial_probs: list[float]
    event_vocab: list[str]
    physx_log_likelihoods: list[float] | None = None


class HypothesisResult(BaseModel):
    rank: int
    description: str
    log_probability: float
    posterior: float
    events: list[str]


class InferResponse(BaseModel):
    status: str
    hypotheses: list[HypothesisResult]


@router.post("/rank", response_model=InferResponse)
async def rank_hypotheses(req: InferRequest) -> InferResponse:
    """Rank candidate event sequences by probabilistic likelihood."""
    if len(req.sequences) != len(req.descriptions):
        raise HTTPException(status_code=400, detail="sequences and descriptions must have equal length")

    try:
        from forensim.infer.sequence import rank_event_sequences, integrate_physx_scores

        hypotheses = rank_event_sequences(
            sequences=req.sequences,
            descriptions=req.descriptions,
            transition_matrix=req.transition_matrix,
            initial_probs=req.initial_probs,
            event_vocab=req.event_vocab,
        )

        if req.physx_log_likelihoods:
            hypotheses = integrate_physx_scores(hypotheses, req.physx_log_likelihoods)

        return InferResponse(
            status="success",
            hypotheses=[
                HypothesisResult(
                    rank=i + 1,
                    description=h.description,
                    log_probability=h.log_probability,
                    posterior=h.posterior,
                    events=h.events,
                )
                for i, h in enumerate(hypotheses)
            ],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
