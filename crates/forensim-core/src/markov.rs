/// Markov Chain for event-sequence scoring.
///
/// Given a transition matrix and an observed event sequence,
/// computes the log-probability of that sequence.

use pyo3::prelude::*;

#[pyclass]
pub struct MarkovChain {
    /// transition[i][j] = P(event j follows event i)
    transition: Vec<Vec<f64>>,
    /// initial[i] = P(sequence starts with event i)
    initial: Vec<f64>,
    n_states: usize,
}

#[pymethods]
impl MarkovChain {
    #[new]
    pub fn new(transition: Vec<Vec<f64>>, initial: Vec<f64>) -> PyResult<Self> {
        let n_states = initial.len();
        if transition.len() != n_states {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "transition matrix row count must match number of states",
            ));
        }
        Ok(Self { transition, initial, n_states })
    }

    /// Learn transition and initial probabilities from observed sequences.
    ///
    /// Uses Laplace (add-1) smoothing so no probability is exactly zero.
    #[staticmethod]
    pub fn fit_from_sequences(
        sequences: Vec<Vec<usize>>,
        n_states: usize,
    ) -> PyResult<MarkovChain> {
        if n_states == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_states must be > 0",
            ));
        }
        // Laplace-smoothed counts
        let mut trans_counts = vec![vec![1.0f64; n_states]; n_states];
        let mut init_counts = vec![1.0f64; n_states];

        for seq in &sequences {
            if seq.is_empty() {
                continue;
            }
            if let Some(&s0) = seq.first() {
                if s0 < n_states {
                    init_counts[s0] += 1.0;
                }
            }
            for w in seq.windows(2) {
                let (i, j) = (w[0], w[1]);
                if i < n_states && j < n_states {
                    trans_counts[i][j] += 1.0;
                }
            }
        }

        let init_total: f64 = init_counts.iter().sum();
        let initial: Vec<f64> = init_counts.iter().map(|c| c / init_total).collect();

        let transition: Vec<Vec<f64>> = trans_counts
            .into_iter()
            .map(|row| {
                let row_total: f64 = row.iter().sum();
                row.into_iter().map(|c| c / row_total).collect()
            })
            .collect();

        Ok(MarkovChain { transition, initial, n_states })
    }

    /// Compute log P(sequence) under this Markov chain.
    pub fn log_probability(&self, sequence: Vec<usize>) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }
        let mut log_p = self.initial.get(sequence[0]).copied().unwrap_or(0.0).ln();
        for w in sequence.windows(2) {
            let (i, j) = (w[0], w[1]);
            let p = self
                .transition
                .get(i)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(0.0);
            log_p += p.ln();
        }
        log_p
    }

    /// Score and rank a list of candidate sequences.
    /// Returns indices sorted best-to-worst.
    pub fn rank_sequences(&self, sequences: Vec<Vec<usize>>) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = sequences
            .iter()
            .enumerate()
            .map(|(idx, seq)| (idx, self.log_probability(seq.clone())))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Return a clone of the transition matrix.
    pub fn transition_matrix(&self) -> Vec<Vec<f64>> {
        self.transition.clone()
    }

    /// Power-iteration stationary distribution: π such that πA = π.
    ///
    /// Runs 1000 iterations from a uniform starting distribution.
    pub fn steady_state(&self) -> Vec<f64> {
        let n = self.n_states;
        if n == 0 {
            return vec![];
        }
        let mut pi = vec![1.0 / n as f64; n];
        for _ in 0..1000 {
            let mut next = vec![0.0f64; n];
            for j in 0..n {
                for i in 0..n {
                    next[j] += pi[i]
                        * self.transition.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                }
            }
            let total: f64 = next.iter().sum();
            if total > 0.0 {
                pi = next.into_iter().map(|v| v / total).collect();
            }
        }
        pi
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn uniform_2x2() -> MarkovChain {
        MarkovChain::new(
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![0.5, 0.5],
        )
        .unwrap()
    }

    #[test]
    fn test_log_probability_empty() {
        let chain = uniform_2x2();
        assert_eq!(chain.log_probability(vec![]), 0.0);
    }

    #[test]
    fn test_log_probability_known() {
        // transition: state0→state1 with p=0.9, state1→state0 with p=0.8
        let chain = MarkovChain::new(
            vec![vec![0.1, 0.9], vec![0.8, 0.2]],
            vec![1.0, 0.0],
        )
        .unwrap();
        // seq [0, 1]: log(1.0) + log(0.9)
        let lp = chain.log_probability(vec![0, 1]);
        assert_relative_eq!(lp, (0.9f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_rank_sequences() {
        let chain = MarkovChain::new(
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.9, 0.1],
        )
        .unwrap();
        // seq [0,0,0] should score higher than [0,1,0]
        let ranked = chain.rank_sequences(vec![vec![0, 1, 0], vec![0, 0, 0]]);
        assert_eq!(ranked[0], 1); // [0,0,0] is better
    }

    #[test]
    fn test_fit_from_sequences_row_sums() {
        let seqs = vec![vec![0usize, 1, 0, 1], vec![0, 0, 1, 1]];
        let chain = MarkovChain::fit_from_sequences(seqs, 2).unwrap();
        for row in chain.transition_matrix() {
            assert_relative_eq!(row.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_steady_state_sums_to_one() {
        let chain = MarkovChain::new(
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            vec![0.5, 0.5],
        )
        .unwrap();
        let ss = chain.steady_state();
        assert_relative_eq!(ss.iter().sum::<f64>(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_transition_matrix_round_trip() {
        let seqs = vec![vec![0usize, 1, 2, 0], vec![1, 2, 0, 1]];
        let chain = MarkovChain::fit_from_sequences(seqs, 3).unwrap();
        let mat = chain.transition_matrix();
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
    }

    #[test]
    fn test_fit_zero_states_errors() {
        assert!(MarkovChain::fit_from_sequences(vec![], 0).is_err());
    }
}
