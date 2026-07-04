# Probabilistic Model Reference

This document is the technical reference for ForenSim's probabilistic inference pipeline. It covers the mathematical foundations, implementation details, and guidance on interpreting outputs.

## Table of Contents

1. [Overview](#1-overview)
2. [Markov Chain Sequence Scoring](#2-markov-chain-sequence-scoring)
3. [Hidden Markov Model Mode](#3-hidden-markov-model-mode)
4. [Bayesian Update Steps](#4-bayesian-update-steps)
5. [Annotation-Driven Likelihoods](#5-annotation-driven-likelihoods)
6. [Monte Carlo Integration from PhysX](#6-monte-carlo-integration-from-physx)
7. [Sensitivity Analysis](#7-sensitivity-analysis)
8. [Interpreting Outputs](#8-interpreting-outputs)
9. [Mathematical Reference](#9-mathematical-reference)

---

## 1. Overview

ForenSim's inference pipeline answers a single question: *given the physical evidence collected at this scene, how probable is each competing hypothesis about what happened?*

The pipeline has four interlocking components:

```
Evidence annotations (ROI tags)
        │
        ▼  forensim.infer.evidence
Log-likelihood vectors  ──────────────────────────────────┐
                                                          │
COLMAP + gsplat reconstruction                            │
        │                                                 │
        ▼  forensim.simulate                              │
PhysX Monte Carlo trajectories                            │
        │                                                 ▼
        │                               forensim-core  (Rust)
        └──────────────────────────►  BayesEngine.update()
                                              │
                                    Markov chain scoring
                                    HMM Viterbi (optional)
                                              │
                                              ▼
                                    Ranked hypotheses
                                    P(H|E), Bayes factors,
                                    Shannon entropy,
                                    LOO sensitivity scores
```

Each component is described below. Formulae use LaTeX-style inline notation: `P(A|B)` denotes the conditional probability of A given B; `log` denotes the natural logarithm unless noted.

---

## 2. Markov Chain Sequence Scoring

### What the transition matrix represents

An event sequence is modelled as a first-order Markov chain over a finite state vocabulary. Each state `s_i` represents a discrete forensic event type, for example:

```
States:  entry | search | confrontation | impact | struggle | exit | none
```

The transition matrix `T` is an `(N × N)` row-stochastic matrix where `T[i][j] = P(s_{t+1} = j | s_t = i)` — the probability that event `j` follows event `i` in a realistic scenario.

ForenSim ships a domain-knowledge prior matrix populated from forensic case literature. It can be overridden per-project:

```python
from forensim._core import MarkovChain
import numpy as np

# 4-state example: entry → confrontation → impact → exit
T = np.array([
    #  entry  confront  impact  exit
    [0.05,   0.70,     0.10,   0.15],  # from entry
    [0.10,   0.15,     0.60,   0.15],  # from confrontation
    [0.05,   0.20,     0.20,   0.55],  # from impact
    [0.50,   0.05,     0.05,   0.40],  # from exit (loop / re-entry)
])

chain = MarkovChain(
    transition_matrix=T.tolist(),
    states=["entry", "confrontation", "impact", "exit"],
)
```

### Sequence scoring

Given a proposed event sequence `S = [s_1, s_2, ..., s_k]`, the Markov log-likelihood is:

```
log P(S | T) = log π(s_1) + Σ_{t=1}^{k-1} log T[s_t][s_{t+1}]
```

where `π` is the initial state distribution (uniform by default).

```python
log_prob = chain.score_sequence(["entry", "confrontation", "impact", "exit"])
# Returns: log P(S | T)  (a negative float; closer to 0 = more probable)
```

### Configuring the transition matrix

Edit the matrix directly as shown above, or load a pre-configured profile:

```python
chain = MarkovChain.from_profile("burglary")        # burglary scenario priors
chain = MarkovChain.from_profile("domestic_assault") # domestic violence priors
chain = MarkovChain.from_profile("vehicle_collision") # traffic accident priors
```

Custom profiles are JSON files stored in `assets/markov-profiles/`. The schema:

```json
{
  "profile": "burglary",
  "states": ["entry", "search", "theft", "confrontation", "exit"],
  "transition": [[...], [...], ...],
  "initial": [0.9, 0.05, 0.02, 0.02, 0.01]
}
```

---

## 3. Hidden Markov Model Mode

### When to use HMM mode

Use the HMM when the true event sequence is not directly observable — only noisy physical observations are available. This is the more general case; Markov scoring (§2) is a special case where observations equal states.

Example: a 3DGS reconstruction shows evidence consistent with multiple events at each timestep. The HMM treats these as *emissions* from hidden states.

| Mode | Use when |
|---|---|
| Markov chain scoring | You have a proposed sequence of events and want to score it |
| HMM Viterbi | You have a sequence of physical observations and want to infer the most likely hidden event sequence |
| HMM forward | You want the total log-probability of an observation sequence under a hypothesis |

### The emission matrix

The emission matrix `E` is an `(N_states × N_observations)` matrix where `E[i][o] = P(observation = o | hidden state = i)`.

Observations are discretized physical measurements: object velocity bins, impact force bins, spatial proximity categories, etc.

```python
from forensim._core import HiddenMarkovModel

hmm = HiddenMarkovModel(
    transition=T.tolist(),    # same as MarkovChain
    emission=E.tolist(),      # N_states × N_obs
    initial=pi.tolist(),      # initial state distribution
)

# Viterbi: find the most likely hidden state sequence
best_path = hmm.viterbi(observed_events)          # List[str]

# Forward: total log-likelihood of the observation sequence
log_prob = hmm.forward_log_likelihood(observed_events)  # float
```

### Viterbi algorithm

The Viterbi algorithm finds the sequence `S*` that maximizes:

```
S* = argmax_S  P(S, O | λ)
   = argmax_S  π(s_1) · E[s_1][o_1] · Π_{t=2}^{T} T[s_{t-1}][s_t] · E[s_t][o_t]
```

ForenSim's implementation operates in log-space to avoid numerical underflow:

```
δ_t(i) = max_{s_{1..t-1}} log P(s_1,..,s_t=i, o_1,..,o_t | λ)

δ_1(i) = log π(i) + log E[i][o_1]
δ_t(i) = max_j (δ_{t-1}(j) + log T[j][i]) + log E[i][o_t]
```

The implementation is in `crates/forensim-core/src/hmm.rs` and runs in `O(T · N²)` time.

---

## 4. Bayesian Update Steps

### The update equation

ForenSim maintains a probability distribution over hypotheses `H = {H_1, H_2, ..., H_K}`. At any point the posterior is:

```
P(H_k | E) ∝ P(E | H_k) · P(H_k)
```

where:
- `P(H_k)` is the prior (uniform by default, or set from a domain profile)
- `P(E | H_k)` is the likelihood of observing evidence E under hypothesis k
- `P(H_k | E)` is the posterior after normalizing over all hypotheses

### Sequential update

When new evidence `e_i` arrives (e.g., a new annotation ROI), the posterior is updated incrementally:

```
P(H_k | e_1, ..., e_i) ∝ P(e_i | H_k) · P(H_k | e_1, ..., e_{i-1})
```

This is sequential Bayesian updating, and it is exactly what `BayesEngine.update()` implements:

```python
from forensim._core import BayesEngine

engine = BayesEngine(
    hypotheses=["entry_first", "confrontation_first", "accident"],
    prior=[0.333, 0.333, 0.334],
)

# Each update call takes a vector of P(evidence | H_k) for each hypothesis
engine.update(likelihoods=[0.8, 0.3, 0.1])   # first piece of evidence
engine.update(likelihoods=[0.6, 0.5, 0.2])   # second piece of evidence

posterior = engine.posterior()   # normalized probabilities, sums to 1.0
bayes_factors = engine.bayes_factors()  # BF_{k,k'} for all pairs
entropy = engine.entropy()       # Shannon entropy H(posterior)
```

### Log-space computation

Internally, ForenSim stores log-probabilities throughout to avoid underflow when combining many independent pieces of evidence:

```
log P(H_k | E) = log P(H_k) + Σ_i log P(e_i | H_k) − log Z
```

where `Z = Σ_k exp(log P(H_k) + Σ_i log P(e_i | H_k))` is the normalization constant computed via the log-sum-exp trick.

---

## 5. Annotation-Driven Likelihoods

### How ROI tags map to log-likelihoods

Each annotation ROI carries a semantic tag from a controlled vocabulary. The `forensim.infer.evidence` module maps each tag to a per-hypothesis log-likelihood contribution `log P(tag | H_k)`.

The default likelihood table (configurable per project):

```python
# forensim/infer/evidence.py
TAG_LOG_LIKELIHOOD_TABLE: dict[str, list[float]] = {
    # tag             H_entry  H_exit   H_struggle  H_accident
    "blood":         [-0.50,   -1.20,   -0.30,      -0.80],
    "impact":        [-0.60,   -0.90,   -0.40,      -0.50],
    "footprint":     [-0.30,   -0.40,   -0.70,      -1.10],
    "trajectory":    [-0.40,   -0.40,   -0.80,      -0.60],
    "entry-point":   [-0.20,   -2.00,   -1.50,      -1.80],
    "exit-point":    [-2.10,   -0.20,   -1.40,      -1.70],
    "weapon":        [-0.70,   -0.80,   -0.25,      -1.50],
    "drag-mark":     [-0.55,   -0.60,   -0.35,      -1.20],
}
```

These values are log-probabilities (natural log). A value of `-0.20` corresponds to `P ≈ 0.82`; a value of `-2.10` corresponds to `P ≈ 0.12`.

### Analyst confidence weighting

Each annotation carries an `analyst_confidence ∈ [0, 1]` set by the reviewing analyst. The effective log-likelihood contribution is:

```
log L_i(H_k) = analyst_confidence_i × log P(tag_i | H_k)
```

A confidence of 1.0 applies the full log-likelihood. A confidence of 0.5 halves the magnitude of the log-likelihood — equivalent to saying the evidence is half as informative. A confidence of 0.0 effectively removes the annotation from inference (log L = 0 → likelihood ratio = 1 → no update).

### Customizing the likelihood table

Override the table in your project configuration:

```python
from forensim.infer.evidence import EvidenceScorer

scorer = EvidenceScorer(
    hypotheses=["entry_first", "confrontation_first", "accident"],
    tag_table={
        "blood": {"entry_first": -0.4, "confrontation_first": -0.2, "accident": -0.9},
        "impact": {"entry_first": -0.5, "confrontation_first": -0.3, "accident": -0.4},
    }
)

log_likelihoods = scorer.score(annotations)  # List[float], one per hypothesis
```

---

## 6. Monte Carlo Integration from PhysX

### Why Monte Carlo?

The likelihood `P(E | H_k)` requires knowing how plausible the physical evidence is given a particular event hypothesis. This cannot be computed analytically for arbitrary rigid-body dynamics. Instead, ForenSim uses Monte Carlo integration:

1. Run `M` PhysX simulations, each with initial conditions sampled from a distribution consistent with hypothesis `H_k`.
2. Each simulation produces a trajectory `τ_j`.
3. Evaluate `P(E | τ_j)` — how well does the simulated trajectory match the observed evidence?
4. Estimate the likelihood by averaging: `P(E | H_k) ≈ (1/M) Σ_{j=1}^{M} P(E | τ_j)`

### Trajectory-to-evidence comparison

`P(E | τ_j)` is computed by comparing each piece of physical evidence against the trajectory:

- **Impact location:** Gaussian likelihood over the distance between the simulated impact point and the observed evidence location.
- **Object velocity:** Gaussian likelihood over the difference between the simulated object velocity at a key event time and the estimated velocity from trajectory analysis.
- **Deformation pattern:** KL divergence between the simulated deformation field and the observed damage pattern.

```python
def evidence_likelihood(evidence: PhysicalEvidence, trajectory: Trajectory) -> float:
    """Returns log P(evidence | trajectory)."""
    impact_loc_loglik = normal_logpdf(
        x=evidence.impact_location,
        mu=trajectory.impact_location,
        sigma=IMPACT_LOCATION_SIGMA,
    )
    velocity_loglik = normal_logpdf(
        x=evidence.estimated_velocity,
        mu=trajectory.velocity_at_impact,
        sigma=VELOCITY_SIGMA,
    )
    return impact_loc_loglik + velocity_loglik
```

### Monte Carlo sampling configuration

```python
from forensim.simulate import MonteCarloRunner

runner = MonteCarloRunner(
    scene_usd="output/crime-scene-01/scene.usda",
    num_scenarios=500,           # number of PhysX runs per hypothesis
    initial_velocity_range=(-5.0, 5.0),  # m/s, sampled uniformly per axis
    friction_range=(0.3, 0.7),   # sampled per surface
    seed=42,                     # for reproducibility
)

results = runner.run(hypothesis="confrontation_first")
# results.trajectories: List[Trajectory]
# results.mean_log_likelihood: float
# results.std_log_likelihood: float
```

With `num_scenarios=500` and a scene containing ~5 rigid bodies, a full Monte Carlo run takes approximately 3–8 minutes on an RTX 3060 Ti.

---

## 7. Sensitivity Analysis

### Leave-one-out methodology

Sensitivity analysis answers: *which piece of evidence most influences the top-ranked hypothesis?*

The procedure is leave-one-out (LOO):

1. Compute the full posterior `P(H | E)` using all `N` annotations.
2. For each annotation `e_i`:
   a. Remove `e_i` from the evidence set.
   b. Recompute the posterior `P(H | E \ {e_i})`.
   c. Record the rank change of the top hypothesis and the shift in its posterior probability.
3. Report the annotation `e_i` whose removal causes the largest change.

```python
from forensim.infer import SensitivityAnalyzer

analyzer = SensitivityAnalyzer(engine=bayes_engine)
results = analyzer.loo(annotations=annotations)

# results is a list of (annotation_id, delta_posterior_top, rank_change)
# sorted by |delta_posterior_top| descending
for ann_id, delta, rank_change in results:
    print(f"{ann_id:20s}  Δ posterior = {delta:+.4f}  rank change = {rank_change:+d}")
```

### Interpreting impact scores

| `|Δ posterior|` | Interpretation |
|---|---|
| < 0.02 | Negligible influence — this annotation barely affects the conclusion |
| 0.02 – 0.10 | Moderate influence — worth verifying |
| 0.10 – 0.25 | Significant influence — this annotation is load-bearing for the conclusion |
| > 0.25 | Dominant influence — the top hypothesis depends heavily on this single piece of evidence; investigate carefully |

A `rank_change != 0` means removing the annotation would change which hypothesis ranks first — the most serious case in a forensic context.

### Impact bar visualization

The ForenSim UI renders LOO results as a horizontal bar chart in the Probability Panel. Each bar represents one annotation; bar length encodes `|Δ posterior|`; color encodes whether removal helps or hurts the top hypothesis.

---

## 8. Interpreting Outputs

### Posterior probability `P(H_k | E)`

The main output: a probability for each competing hypothesis after all evidence has been incorporated. Values sum to 1.0 across hypotheses.

```json
{
  "hypotheses": [
    { "id": "confrontation_first", "posterior": 0.612 },
    { "id": "entry_first",         "posterior": 0.284 },
    { "id": "accident",            "posterior": 0.104 }
  ]
}
```

A posterior of 0.612 means that, given this evidence and these priors, hypothesis `confrontation_first` is about 2.2× more probable than `entry_first`. This is not a statement of absolute truth — it is a relative probability under the stated model assumptions.

### Bayes Factor `BF_{k,k'}`

The Bayes factor compares two hypotheses directly, independent of the prior:

```
BF_{k,k'} = P(E | H_k) / P(E | H_{k'})
```

A `BF > 1` favors `H_k`; `BF < 1` favors `H_{k'}`.

Standard interpretive scale (Jeffreys, 1961):

| `log10(BF)` | Strength of evidence |
|---|---|
| 0 – 0.5 | Barely worth mentioning |
| 0.5 – 1.0 | Substantial |
| 1.0 – 2.0 | Strong |
| > 2.0 | Decisive |

The Bayes factor is prior-independent, making it the preferred statistic for formal forensic reporting. ForenSim reports `BF_{top, second}` — the Bayes factor of the top hypothesis over the second-ranked hypothesis.

### Shannon entropy `H(posterior)`

```
H = −Σ_k P(H_k | E) · log P(H_k | E)
```

Entropy measures how concentrated the posterior is across hypotheses.

| `H` | Interpretation |
|---|---|
| 0.0 | Certainty — all probability mass on one hypothesis |
| `log(K)` (maximum) | Complete uncertainty — all hypotheses equally probable |
| Intermediate | Partial evidence has ruled out some hypotheses |

For `K = 3` hypotheses, maximum entropy = `log(3) ≈ 1.099`. An entropy of 0.3 would indicate strong concentration toward one hypothesis.

### Credible intervals

The PyMC posterior sampler additionally reports 94% credible intervals (HDI — Highest Density Interval) for each hypothesis probability. These are shown in the UI as error bars on the hypothesis list.

---

## 9. Mathematical Reference

### Notation

| Symbol | Meaning |
|---|---|
| `H_k` | Hypothesis k (one of K competing event sequences) |
| `E = {e_1, ..., e_N}` | Set of N evidence items (annotations + trajectories) |
| `P(H_k)` | Prior probability of hypothesis k |
| `P(E | H_k)` | Likelihood of evidence given hypothesis k |
| `P(H_k | E)` | Posterior probability of hypothesis k |
| `T` | Markov transition matrix, shape (N_states × N_states) |
| `E_mat` | HMM emission matrix, shape (N_states × N_obs) |
| `π` | Initial state distribution, shape (N_states,) |
| `τ_j` | Trajectory j from PhysX Monte Carlo simulation |
| `M` | Number of Monte Carlo trajectories per hypothesis |
| `δ_t(i)` | Viterbi log-probability at time t for state i |
| `BF_{k,k'}` | Bayes factor comparing H_k to H_{k'} |
| `H(p)` | Shannon entropy of distribution p |

### Full inference formula

Combining Markov sequence likelihood, annotation likelihoods, and Monte Carlo trajectory likelihoods:

```
log P(H_k | E) ∝ log P(H_k)
               + log P_markov(S_k | T)                    [sequence prior]
               + Σ_i c_i · log P(tag_i | H_k)            [annotation evidence]
               + log [ (1/M) Σ_j P(E_phys | τ_j(H_k)) ]  [physics likelihood]
```

where:
- `S_k` is the canonical event sequence for hypothesis k
- `c_i = analyst_confidence_i ∈ [0,1]` is the confidence weight for annotation i
- `τ_j(H_k)` is the j-th PhysX trajectory sampled under hypothesis k
- `E_phys` is the physical evidence (impact locations, deformation, velocity)

The normalization constant `Z = Σ_k exp(log P(H_k | E)_unnormalized)` is computed via log-sum-exp for numerical stability.

### Viterbi recurrence (log-space)

```
Initialization:
  δ_1(i) = log π(i) + log E_mat[i][o_1]
  ψ_1(i) = 0

Recursion (t = 2, ..., T):
  δ_t(i) = max_{j} (δ_{t-1}(j) + log T[j][i]) + log E_mat[i][o_t]
  ψ_t(i) = argmax_{j} (δ_{t-1}(j) + log T[j][i])

Termination:
  P* = max_i δ_T(i)
  s*_T = argmax_i δ_T(i)

Backtracking:
  s*_t = ψ_{t+1}(s*_{t+1})   for t = T-1, ..., 1
```

### Monte Carlo log-likelihood (log-sum-exp)

```
log P(E | H_k) ≈ log [ (1/M) Σ_j exp(log P(E | τ_j)) ]
              = log(1/M) + logsumexp_j(log P(E | τ_j))
              = -log(M) + logsumexp_j(log P(E | τ_j))
```

This is numerically stable and avoids summing tiny probabilities directly.

### Bayes factor from log-likelihoods

```
log BF_{k,k'} = log P(E | H_k) − log P(E | H_{k'})
BF_{k,k'} = exp(log BF_{k,k'})
```

### Shannon entropy

```
H(p) = −Σ_{k=1}^{K} p_k · log p_k
```

ForenSim uses natural logarithm (`log_e`) throughout; entropy is therefore in nats. To convert to bits, divide by `log(2) ≈ 0.693`.
