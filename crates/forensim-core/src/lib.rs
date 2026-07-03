/// forensim-core: Performance-critical probabilistic engine.
///
/// This crate provides:
///   - Markov chain sequence scoring
///   - Hidden Markov Model (Viterbi, forward-backward)
///   - Monte Carlo Bayesian inference
///   - Statistical utilities (distributions, summary stats)
///
/// Exposed to Python via PyO3 as `forensim._core`.

use pyo3::prelude::*;

mod bayes;
mod hmm;
mod markov;
mod monte_carlo;
mod stats;

/// Register all sub-modules with the Python extension.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<hmm::HiddenMarkovModel>()?;
    m.add_class::<markov::MarkovChain>()?;
    m.add_class::<bayes::BayesianUpdater>()?;
    m.add_class::<monte_carlo::MonteCarloEngine>()?;
    m.add_function(wrap_pyfunction!(stats::summary_stats, m)?)?;
    m.add_function(wrap_pyfunction!(stats::log_normal_pdf, m)?)?;
    Ok(())
}
