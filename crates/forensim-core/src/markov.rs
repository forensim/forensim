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

    pub fn n_states(&self) -> usize {
        self.n_states
    }
}
