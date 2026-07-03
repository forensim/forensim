/// Statistical utility functions exposed to Python.

use pyo3::prelude::*;
use std::f64::consts::PI;

/// Compute summary statistics for a list of values.
/// Returns a dict: { mean, variance, std_dev, min, max, median }.
#[pyfunction]
pub fn summary_stats(values: Vec<f64>) -> PyResult<std::collections::HashMap<String, f64>> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("values cannot be empty"));
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut map = std::collections::HashMap::new();
    map.insert("mean".into(), mean);
    map.insert("variance".into(), variance);
    map.insert("std_dev".into(), std_dev);
    map.insert("min".into(), min);
    map.insert("max".into(), max);
    map.insert("median".into(), median);
    Ok(map)
}

/// Log-PDF of the normal distribution N(mu, sigma) at x.
#[pyfunction]
pub fn log_normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - (2.0 * PI).sqrt().ln()
}
