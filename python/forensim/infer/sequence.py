"""
Event sequence likelihood computation.

Bridges the Rust forensim._core probabilistic engine with the
Python simulation pipeline. Scores and ranks candidate event sequences
using Markov chains, HMMs, and Monte Carlo importance weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forensim.annotate.manager import Annotation
from forensim.annotate.weights import apply_annotation_likelihoods


@dataclass
class ScoredHypothesis:
    """A ranked forensic event hypothesis."""
    index: int
    description: str
    log_probability: float
    posterior: float
    events: list[str]
    bayes_factor: float = float("nan")
    """Bayes factor vs the top-ranked hypothesis (1.0 for rank-1)."""
    entropy: float = float("nan")
    """Shannon entropy of the posterior distribution at the time this was scored."""
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_core() -> Any:
    """Import forensim._core, raising a clear error if not built."""
    try:
        import forensim._core as core  # type: ignore[import-untyped]
        return core
    except ImportError as e:
        raise ImportError(
            "forensim._core not built. Run: maturin develop --release"
        ) from e


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
    core = _import_core()
    chain = core.MarkovChain(transition_matrix, initial_probs)
    vocab_index = {name: i for i, name in enumerate(event_vocab)}

    scored: list[ScoredHypothesis] = []
    for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
        int_seq = [vocab_index[e] for e in seq if e in vocab_index]
        log_p = chain.log_probability(int_seq)
        scored.append(ScoredHypothesis(
            index=idx,
            description=desc,
            log_probability=log_p,
            posterior=float("nan"),
            events=seq,
        ))

    scored = _normalise_posteriors(scored)
    scored.sort(key=lambda h: h.log_probability, reverse=True)
    _annotate_bayes_factors(scored)
    return scored


def rank_event_sequences_hmm(
    sequences: list[list[str]],
    descriptions: list[str],
    transition_matrix: list[list[float]],
    emission_matrix: list[list[float]],
    initial_probs: list[float],
    event_vocab: list[str],
) -> list[ScoredHypothesis]:
    """
    Score and rank candidate event sequences using a Hidden Markov Model.

    Uses the forward algorithm log-likelihood to score each observation
    sequence against the HMM, then ranks hypotheses best-to-worst.

    Args:
        sequences:        List of observation sequences (event names).
        descriptions:     Human-readable description per sequence.
        transition_matrix: HMM state transition matrix [n_states × n_states].
        emission_matrix:   HMM emission matrix [n_states × n_obs].
        initial_probs:     Initial state distribution.
        event_vocab:       Event name → observation index mapping.

    Returns:
        List of ScoredHypothesis sorted best-to-worst.
    """
    core = _import_core()
    hmm = core.HiddenMarkovModel(transition_matrix, emission_matrix, initial_probs)
    vocab_index = {name: i for i, name in enumerate(event_vocab)}

    scored: list[ScoredHypothesis] = []
    for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
        int_seq = [vocab_index[e] for e in seq if e in vocab_index]
        log_p = hmm.forward_log_likelihood(int_seq)
        scored.append(ScoredHypothesis(
            index=idx,
            description=desc,
            log_probability=log_p,
            posterior=float("nan"),
            events=seq,
        ))

    scored = _normalise_posteriors(scored)
    scored.sort(key=lambda h: h.log_probability, reverse=True)
    _annotate_bayes_factors(scored)
    return scored


def integrate_physx_scores(
    hypotheses: list[ScoredHypothesis],
    physx_log_likelihoods: list[float],
) -> list[ScoredHypothesis]:
    """
    Update hypothesis posteriors by integrating PhysX simulation likelihoods.

    Multiplies sequence prior by physics likelihood (log-space addition),
    re-normalises posteriors, and re-ranks.

    Args:
        hypotheses:             Previously scored hypotheses.
        physx_log_likelihoods:  log P(evidence | hypothesis_i) from PhysX MC.

    Returns:
        Updated and re-ranked hypotheses.
    """
    if len(hypotheses) != len(physx_log_likelihoods):
        raise ValueError("hypothesis count must match physx_log_likelihoods count")

    combined = (
        np.array([h.log_probability for h in hypotheses])
        + np.array(physx_log_likelihoods)
    )
    for h, lp in zip(hypotheses, combined):
        h.log_probability = float(lp)

    hypotheses = _normalise_posteriors(hypotheses)
    hypotheses.sort(key=lambda h: h.log_probability, reverse=True)
    _annotate_bayes_factors(hypotheses)
    return hypotheses


def bayesian_update(
    hypotheses: list[ScoredHypothesis],
    likelihoods: list[float],
) -> list[ScoredHypothesis]:
    """
    Run a Bayesian update step using the Rust BayesianUpdater.

    Uses the current posteriors as priors and updates with the given likelihoods.

    Args:
        hypotheses:  Current hypothesis set (posteriors used as priors).
        likelihoods: P(new_evidence | hypothesis_i) — raw (not log) likelihoods.

    Returns:
        Updated hypotheses re-ranked by posterior.
    """
    core = _import_core()
    priors = [h.posterior for h in hypotheses]
    # Ensure priors are valid probabilities
    priors_arr = np.array(priors, dtype=float)
    priors_arr = np.clip(priors_arr, 1e-12, None)
    priors_arr /= priors_arr.sum()

    updater = core.BayesianUpdater(priors_arr.tolist())
    new_posteriors = updater.update(likelihoods)
    entropy = updater.entropy()

    for h, p in zip(hypotheses, new_posteriors):
        h.posterior = float(p)
        h.entropy = entropy

    hypotheses.sort(key=lambda h: h.posterior, reverse=True)
    _annotate_bayes_factors(hypotheses)
    return hypotheses


def integrate_annotation_scores(
    hypotheses: list[ScoredHypothesis],
    annotations: list[Annotation],
    strength: float = 1.0,
) -> list[ScoredHypothesis]:
    """Update hypothesis scores by integrating evidence annotation likelihoods.

    Args:
        hypotheses: Previously scored hypotheses.
        annotations: Evidence ROI annotations with forensic tags.
        strength: Scaling factor for the annotation likelihood effect.

    Returns:
        Updated and re-ranked hypotheses.
    """
    if not annotations:
        return hypotheses

    log_likelihoods = apply_annotation_likelihoods(hypotheses, annotations, strength)
    combined = np.array([h.log_probability for h in hypotheses]) + np.array(log_likelihoods)
    for h, lp in zip(hypotheses, combined):
        h.log_probability = float(lp)

    hypotheses = _normalise_posteriors(hypotheses)
    hypotheses.sort(key=lambda h: h.log_probability, reverse=True)
    _annotate_bayes_factors(hypotheses)
    return hypotheses


def compute_posterior_entropy(posteriors: list[float]) -> float:
    """Shannon entropy of a probability distribution in nats."""
    arr = np.array(posteriors, dtype=float)
    arr = arr[arr > 0]
    return float(-np.sum(arr * np.log(arr)))


# ── Private helpers ────────────────────────────────────────────────────────────

def _normalise_posteriors(hypotheses: list[ScoredHypothesis]) -> list[ScoredHypothesis]:
    """Log-sum-exp normalisation to convert log-probs to posteriors."""
    log_probs = np.array([h.log_probability for h in hypotheses])
    # Replace -inf with a very negative finite number for numerical stability
    log_probs = np.where(np.isfinite(log_probs), log_probs, -1e300)
    max_lp = log_probs.max()
    weights = np.exp(log_probs - max_lp)
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(hypotheses)) / len(hypotheses)
    else:
        weights /= total

    entropy = compute_posterior_entropy(weights.tolist())
    for h, w in zip(hypotheses, weights):
        h.posterior = float(w)
        h.entropy = entropy

    return hypotheses


def _annotate_bayes_factors(hypotheses: list[ScoredHypothesis]) -> None:
    """
    Annotate each hypothesis with its Bayes factor vs the top-ranked one.

    BF(H_i vs H_1) = exp(log_P(H_i) - log_P(H_1))

    Assumes hypotheses are already sorted best-to-worst.
    """
    if not hypotheses:
        return
    best_log_p = hypotheses[0].log_probability
    for h in hypotheses:
        if math.isfinite(h.log_probability) and math.isfinite(best_log_p):
            h.bayes_factor = math.exp(h.log_probability - best_log_p)
        else:
            h.bayes_factor = float("nan")
    hypotheses[0].bayes_factor = 1.0
