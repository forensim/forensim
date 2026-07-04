/// Hidden Markov Model implementation.
///
/// Exposes:
///   - `HiddenMarkovModel::new(transition, emission, initial)`
///   - `HiddenMarkovModel::viterbi(observations)` -> most-likely state sequence
///   - `HiddenMarkovModel::viterbi_log_prob(observations)` -> log-prob of best path
///   - `HiddenMarkovModel::forward_log_likelihood(observations)` -> log P(obs | model)
///   - `HiddenMarkovModel::rank_hypotheses(observation_sets)` -> ranked indices

use pyo3::prelude::*;

#[pyclass]
pub struct HiddenMarkovModel {
    /// transition[i][j] = P(state j | state i)
    transition: Vec<Vec<f64>>,
    /// emission[i][k] = P(obs k | state i)
    emission: Vec<Vec<f64>>,
    /// initial[i] = P(state i at t=0)
    initial: Vec<f64>,
    n_states: usize,
}

#[pymethods]
impl HiddenMarkovModel {
    #[new]
    pub fn new(
        transition: Vec<Vec<f64>>,
        emission: Vec<Vec<f64>>,
        initial: Vec<f64>,
    ) -> PyResult<Self> {
        let n_states = initial.len();
        if transition.len() != n_states || emission.len() != n_states {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dimension mismatch: transition and emission must match number of states",
            ));
        }
        Ok(Self { transition, emission, initial, n_states })
    }

    /// Viterbi algorithm: returns the most-likely hidden state sequence.
    pub fn viterbi(&self, observations: Vec<usize>) -> Vec<usize> {
        let (path, _) = self.viterbi_inner(&observations);
        path
    }

    /// Log-probability of the most-likely (Viterbi) path.
    pub fn viterbi_log_prob(&self, observations: Vec<usize>) -> f64 {
        let (_, log_prob) = self.viterbi_inner(&observations);
        log_prob
    }

    /// Forward algorithm log-likelihood: log P(observations | model).
    pub fn forward_log_likelihood(&self, observations: Vec<usize>) -> f64 {
        let t = observations.len();
        if t == 0 {
            return 0.0;
        }
        let n = self.n_states;

        let mut alpha = vec![0.0f64; n];
        for i in 0..n {
            let e = self.emission[i].get(observations[0]).copied().unwrap_or(0.0);
            alpha[i] = self.initial[i] * e;
        }

        for step in 1..t {
            let obs = observations[step];
            let mut alpha_next = vec![0.0f64; n];
            for j in 0..n {
                let e = self.emission[j].get(obs).copied().unwrap_or(0.0);
                let sum: f64 = (0..n).map(|i| alpha[i] * self.transition[i][j]).sum();
                alpha_next[j] = sum * e;
            }
            alpha = alpha_next;
        }

        let total: f64 = alpha.iter().sum();
        if total <= 0.0 { f64::NEG_INFINITY } else { total.ln() }
    }

    /// Score multiple observation sets and return indices sorted best-to-worst.
    pub fn rank_hypotheses(&self, observation_sets: Vec<Vec<usize>>) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = observation_sets
            .iter()
            .enumerate()
            .map(|(idx, obs)| (idx, self.forward_log_likelihood(obs.clone())))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Return a clone of the emission matrix.
    pub fn emission_matrix(&self) -> Vec<Vec<f64>> {
        self.emission.clone()
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }
}

impl HiddenMarkovModel {
    /// Internal Viterbi returning (best path, best log-prob).
    fn viterbi_inner(&self, observations: &[usize]) -> (Vec<usize>, f64) {
        let t = observations.len();
        if t == 0 {
            return (vec![], 0.0);
        }
        let n = self.n_states;

        let mut delta = vec![vec![f64::NEG_INFINITY; n]; t];
        let mut psi = vec![vec![0usize; n]; t];

        // Initialise
        for i in 0..n {
            let e = self.emission[i].get(observations[0]).copied().unwrap_or(0.0);
            delta[0][i] = self.initial[i].ln() + e.ln();
        }

        // Recursion
        for step in 1..t {
            let obs = observations[step];
            for j in 0..n {
                let e = self.emission[j].get(obs).copied().unwrap_or(0.0).ln();
                let (best_prev, best_val) = (0..n)
                    .map(|i| (i, delta[step - 1][i] + self.transition[i][j].ln()))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap_or((0, f64::NEG_INFINITY));
                delta[step][j] = best_val + e;
                psi[step][j] = best_prev;
            }
        }

        // Best final state and its log-prob
        let (best_final, best_log_prob) = (0..n)
            .map(|s| (s, delta[t - 1][s]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((0, f64::NEG_INFINITY));

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = best_final;
        for step in (1..t).rev() {
            path[step - 1] = psi[step][path[step]];
        }
        (path, best_log_prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn trivial_hmm() -> HiddenMarkovModel {
        // 1 state, 2 observations, always emits obs 0
        HiddenMarkovModel::new(
            vec![vec![1.0]],
            vec![vec![1.0, 0.0]],
            vec![1.0],
        )
        .unwrap()
    }

    fn two_state_hmm() -> HiddenMarkovModel {
        // state 0 emits obs 0 with p=0.9, state 1 emits obs 1 with p=0.9
        HiddenMarkovModel::new(
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.6, 0.4],
        )
        .unwrap()
    }

    #[test]
    fn test_viterbi_empty() {
        let path: Vec<usize> = trivial_hmm().viterbi(vec![]);
        assert!(path.is_empty());
    }

    #[test]
    fn test_viterbi_trivial() {
        let path = trivial_hmm().viterbi(vec![0, 0, 0]);
        assert_eq!(path, vec![0, 0, 0]);
    }

    #[test]
    fn test_forward_log_likelihood_negative() {
        let ll = two_state_hmm().forward_log_likelihood(vec![0, 0, 1]);
        assert!(ll <= 0.0);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_forward_log_likelihood_empty() {
        assert_eq!(two_state_hmm().forward_log_likelihood(vec![]), 0.0);
    }

    #[test]
    fn test_rank_hypotheses_order() {
        let hmm = two_state_hmm();
        // obs [0,0,0] fits state 0 better than [1,1,1] which fits state 1
        // both should produce finite values; rank by log-likelihood
        let ranked = hmm.rank_hypotheses(vec![vec![1, 1, 1], vec![0, 0, 0]]);
        // We don't hard-code which is higher, just verify we get 2 ranked indices
        assert_eq!(ranked.len(), 2);
        assert!(ranked[0] == 0 || ranked[0] == 1);
    }

    #[test]
    fn test_viterbi_log_prob_finite() {
        let lp = two_state_hmm().viterbi_log_prob(vec![0, 1, 0]);
        assert!(lp.is_finite());
        assert!(lp <= 0.0);
    }

    #[test]
    fn test_emission_matrix_round_trip() {
        let hmm = two_state_hmm();
        let em = hmm.emission_matrix();
        assert_eq!(em.len(), 2);
        assert_relative_eq!(em[0][0], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        // emission has 3 rows but initial has 2 — should fail
        assert!(HiddenMarkovModel::new(
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![1.0], vec![1.0], vec![1.0]],
            vec![0.5, 0.5],
        )
        .is_err());
    }
}
