/// Hidden Markov Model implementation.
///
/// Exposes:
///   - `HiddenMarkovModel::new(transition, emission, initial)`
///   - `HiddenMarkovModel::viterbi(observations)` -> most-likely state sequence
///   - `HiddenMarkovModel::forward_log_likelihood(observations)` -> log P(obs | model)

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
        let t = observations.len();
        if t == 0 {
            return vec![];
        }
        let n = self.n_states;

        // delta[t][i] = max log-prob of any path ending in state i at time t
        let mut delta = vec![vec![f64::NEG_INFINITY; n]; t];
        // psi[t][i] = argmax predecessor state
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

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = (0..n)
            .max_by(|&a, &b| delta[t - 1][a].partial_cmp(&delta[t - 1][b]).unwrap())
            .unwrap_or(0);
        for step in (1..t).rev() {
            path[step - 1] = psi[step][path[step]];
        }
        path
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

    pub fn n_states(&self) -> usize {
        self.n_states
    }
}
