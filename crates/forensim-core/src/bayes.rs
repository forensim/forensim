/// Bayesian evidence updater.
///
/// Implements iterative Bayesian updating:
///   posterior ∝ likelihood × prior
/// Normalised after each update so values remain probabilities.

use pyo3::prelude::*;

#[pyclass]
pub struct BayesianUpdater {
    /// Prior probabilities over hypotheses (must sum to 1).
    priors: Vec<f64>,
    n_hypotheses: usize,
}

#[pymethods]
impl BayesianUpdater {
    #[new]
    pub fn new(priors: Vec<f64>) -> PyResult<Self> {
        if priors.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("priors cannot be empty"));
        }
        let sum: f64 = priors.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("priors must sum to 1.0, got {sum:.6}"),
            ));
        }
        let n_hypotheses = priors.len();
        Ok(Self { priors, n_hypotheses })
    }

    /// Update posteriors given likelihoods P(evidence | hypothesis_i).
    ///
    /// `likelihoods` must have the same length as the hypothesis count.
    /// Returns the new posterior distribution.
    pub fn update(&mut self, likelihoods: Vec<f64>) -> PyResult<Vec<f64>> {
        if likelihoods.len() != self.n_hypotheses {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "likelihoods length must match number of hypotheses",
            ));
        }
        let unnorm: Vec<f64> = self.priors.iter()
            .zip(likelihoods.iter())
            .map(|(p, l)| p * l)
            .collect();
        let total: f64 = unnorm.iter().sum();
        if total <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All likelihoods are zero — cannot update",
            ));
        }
        self.priors = unnorm.iter().map(|v| v / total).collect();
        Ok(self.priors.clone())
    }

    /// Current posterior (= prior before first update).
    pub fn posteriors(&self) -> Vec<f64> {
        self.priors.clone()
    }

    /// Index of the maximum-a-posteriori (MAP) hypothesis.
    pub fn map_hypothesis(&self) -> usize {
        self.priors
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute Bayes factor: P(E|H_i) / P(E|H_j).
    /// Equivalent to posterior ratio / prior ratio.
    #[staticmethod]
    pub fn bayes_factor(
        prior_i: f64,
        prior_j: f64,
        posterior_i: f64,
        posterior_j: f64,
    ) -> f64 {
        if prior_i <= 0.0 || prior_j <= 0.0 || posterior_j <= 0.0 {
            return f64::NAN;
        }
        (posterior_i / posterior_j) / (prior_i / prior_j)
    }

    pub fn n_hypotheses(&self) -> usize {
        self.n_hypotheses
    }
}
