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

    /// Update using log-likelihoods (avoids underflow via log-sum-exp).
    ///
    /// log_likelihoods[i] = log P(evidence | hypothesis_i)
    /// Returns the new posterior distribution.
    pub fn update_log(&mut self, log_likelihoods: Vec<f64>) -> PyResult<Vec<f64>> {
        if log_likelihoods.len() != self.n_hypotheses {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "log_likelihoods length must match number of hypotheses",
            ));
        }
        // log_unnorm[i] = log(prior[i]) + log_likelihood[i]
        let log_unnorm: Vec<f64> = self.priors.iter()
            .zip(log_likelihoods.iter())
            .map(|(p, ll)| p.ln() + ll)
            .collect();

        // log-sum-exp normalisation
        let max_val = log_unnorm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val.is_infinite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All log-likelihoods are -infinity — cannot update",
            ));
        }
        let log_total = max_val + log_unnorm.iter()
            .map(|&v| (v - max_val).exp())
            .sum::<f64>()
            .ln();

        self.priors = log_unnorm.iter().map(|&v| (v - log_total).exp()).collect();
        Ok(self.priors.clone())
    }

    /// Replace the current priors with new values.
    pub fn reset(&mut self, new_priors: Vec<f64>) -> PyResult<()> {
        if new_priors.len() != self.n_hypotheses {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "new_priors length must match number of hypotheses",
            ));
        }
        let sum: f64 = new_priors.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("new_priors must sum to 1.0, got {sum:.6}"),
            ));
        }
        self.priors = new_priors;
        Ok(())
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

    /// Shannon entropy of the current posterior in nats: −∑ p·ln(p).
    pub fn entropy(&self) -> f64 {
        self.priors.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Bayes factor: P(E|H_i) / P(E|H_j) = (posterior_i/prior_i) / (posterior_j/prior_j).
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
        (posterior_i / prior_i) / (posterior_j / prior_j)
    }

    /// Likelihood ratio: convenience alias for bayes_factor.
    #[staticmethod]
    pub fn likelihood_ratio(
        posterior_i: f64,
        prior_i: f64,
        posterior_j: f64,
        prior_j: f64,
    ) -> f64 {
        if prior_i <= 0.0 || prior_j <= 0.0 || posterior_j <= 0.0 {
            return f64::NAN;
        }
        (posterior_i / prior_i) / (posterior_j / prior_j)
    }

    pub fn n_hypotheses(&self) -> usize {
        self.n_hypotheses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn uniform3() -> BayesianUpdater {
        BayesianUpdater::new(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]).unwrap()
    }

    #[test]
    fn test_new_invalid_sum() {
        assert!(BayesianUpdater::new(vec![0.4, 0.4]).is_err());
    }

    #[test]
    fn test_new_empty() {
        assert!(BayesianUpdater::new(vec![]).is_err());
    }

    #[test]
    fn test_update_normalises() {
        let mut b = uniform3();
        let post = b.update(vec![0.9, 0.05, 0.05]).unwrap();
        assert_relative_eq!(post.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_map_hypothesis() {
        let mut b = uniform3();
        b.update(vec![0.9, 0.05, 0.05]).unwrap();
        assert_eq!(b.map_hypothesis(), 0);
    }

    #[test]
    fn test_entropy_uniform() {
        let b = uniform3();
        // H(uniform 3) = ln(3)
        assert_relative_eq!(b.entropy(), (3.0f64).ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_update_log_matches_update() {
        let likelihoods: Vec<f64> = vec![0.8, 0.1, 0.1];
        let log_likelihoods: Vec<f64> = likelihoods.iter().map(|&l| l.ln()).collect();

        let mut b1 = uniform3();
        let post1 = b1.update(likelihoods).unwrap();

        let mut b2 = uniform3();
        let post2 = b2.update_log(log_likelihoods).unwrap();

        for (p1, p2) in post1.iter().zip(post2.iter()) {
            assert_relative_eq!(p1, p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_likelihood_ratio() {
        // prior = [0.5, 0.5], posterior = [0.8, 0.2]
        // BF = (0.8/0.5) / (0.2/0.5) = 1.6 / 0.4 = 4.0
        let bf = BayesianUpdater::likelihood_ratio(0.8, 0.5, 0.2, 0.5);
        assert_relative_eq!(bf, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut b = uniform3();
        b.reset(vec![0.6, 0.3, 0.1]).unwrap();
        let posts = b.posteriors();
        assert_relative_eq!(posts[0], 0.6, epsilon = 1e-10);
    }

    #[test]
    fn test_reset_wrong_length_errors() {
        let mut b = uniform3();
        assert!(b.reset(vec![0.5, 0.5]).is_err());
    }
}
