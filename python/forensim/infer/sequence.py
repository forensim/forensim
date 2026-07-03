"""
Event sequence likelihood computation.

Bridges the Rust forensim._core probabilistic engine with the
Python simulation pipeline. Scores and ranks candidate event sequences
using Markov chains, HMMs, and Monte Carlo importance weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ScoredHypothesis:
    """A ranked forensic event hypothesis."""
    index: int
    description: str
    log_probability: float
    posterior: float
    events: list[str]
    metadata: dict[str, Any]


def rank_event_sequences(
    sequences: list[list[str]],
    descriptions: list[str],
    transition_matrix: list[list[float]],
    initial_probs: list[float],
    event_vocab: list[str],
) -> list[ScoredHypothesis]:
    """
    Score and rank candidate event sequences using a Markov chain.

    Args:
        sequences:         List of event sequences (each is a list of event names).
        descriptions:      Human-readable description per sequence.
        transition_matrix: P(event_j | event_i) matrix.
        initial_probs:     P(first event = i).
        event_vocab:       Ordered list of event names (index → name mapping).

    Returns:
        List of ScoredHypothesis, sorted best-to-worst.
    """
    try:
        from forensim._core import MarkovChain  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "forensim._core not built. Run: maturin develop --release"
        ) from e

    chain = MarkovChain(transition_matrix, initial_probs)
    vocab_index = {name: i for i, name in enumerate(event_vocab)}

    scored: list[ScoredHypothesis] = []
    for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
        int_seq = [vocab_index[e] for e in seq if e in vocab_index]
        log_p = chain.log_probability(int_seq)
        scored.append(ScoredHypothesis(
            index=idx,
            description=desc,
            log_probability=log_p,
            posterior=float("nan"),  # filled after normalisation
            events=seq,
            metadata={},
        ))

    # Normalise to get approximate posterior (uniform prior)
    log_probs = np.array([h.log_probability for h in scored])
    max_lp = log_probs.max()
    weights = np.exp(log_probs - max_lp)
    weights /= weights.sum()
    for h, w in zip(scored, weights):
        h.posterior = float(w)

    scored.sort(key=lambda h: h.log_probability, reverse=True)
    return scored


def integrate_physx_scores(
    hypotheses: list[ScoredHypothesis],
    physx_log_likelihoods: list[float],
) -> list[ScoredHypothesis]:
    """
    Update hypothesis posteriors by integrating PhysX simulation likelihoods.

    Multiplies sequence prior by physics likelihood (log-space addition).

    Args:
        hypotheses:             Previously scored hypotheses (from rank_event_sequences).
        physx_log_likelihoods:  log P(evidence | hypothesis_i) from PhysX MC.

    Returns:
        Updated and re-ranked hypotheses.
    """
    if len(hypotheses) != len(physx_log_likelihoods):
        raise ValueError("hypothesis count must match physx_log_likelihoods count")

    combined = np.array([h.log_probability for h in hypotheses]) + np.array(physx_log_likelihoods)
    max_lp = combined.max()
    weights = np.exp(combined - max_lp)
    weights /= weights.sum()

    for h, lp, w in zip(hypotheses, combined, weights):
        h.log_probability = float(lp)
        h.posterior = float(w)

    hypotheses.sort(key=lambda h: h.log_probability, reverse=True)
    return hypotheses
