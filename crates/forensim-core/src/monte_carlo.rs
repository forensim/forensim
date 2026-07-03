/// Monte Carlo engine for physics-simulation-based inference.
///
/// Stores a set of simulation trajectories and scores them
/// against observed physical evidence to approximate
/// P(evidence | hypothesis) via importance sampling.

use pyo3::prelude::*;
use rayon::prelude::*;

/// A single simulation trajectory: sequence of (x, y, z) positions per timestep.
type Trajectory = Vec<[f64; 3]>;

#[pyclass]
pub struct MonteCarloEngine {
    /// Stored simulation trajectories (one per scenario).
    trajectories: Vec<Trajectory>,
    /// Observed evidence positions (e.g., final rest positions of objects).
    evidence_positions: Vec<[f64; 3]>,
    /// Gaussian noise sigma for position matching (metres).
    position_sigma: f64,
}

#[pymethods]
impl MonteCarloEngine {
    #[new]
    pub fn new(evidence_positions: Vec<[f64; 3]>, position_sigma: f64) -> Self {
        Self {
            trajectories: Vec::new(),
            evidence_positions,
            position_sigma,
        }
    }

    /// Add a simulation trajectory.
    /// `trajectory` is a flat list of [x, y, z] triples.
    pub fn add_trajectory(&mut self, trajectory: Vec<[f64; 3]>) {
        self.trajectories.push(trajectory);
    }

    /// Clear all stored trajectories.
    pub fn clear_trajectories(&mut self) {
        self.trajectories.clear();
    }

    /// Compute log-likelihood of each trajectory matching the evidence.
    ///
    /// Uses a Gaussian position model: for each evidence point,
    /// find the closest trajectory point and score it.
    /// Returns one log-likelihood per stored trajectory.
    pub fn score_trajectories(&self) -> Vec<f64> {
        let sigma = self.position_sigma;
        let evidence = &self.evidence_positions;

        self.trajectories
            .par_iter()
            .map(|traj| {
                if traj.is_empty() {
                    return f64::NEG_INFINITY;
                }
                let last = traj.last().unwrap();
                let mut log_p = 0.0f64;
                for ev in evidence.iter() {
                    let d2 = (0..3).map(|k| (last[k] - ev[k]).powi(2)).sum::<f64>();
                    // log of Gaussian probability (unnormalised)
                    log_p += -d2 / (2.0 * sigma * sigma);
                }
                log_p
            })
            .collect()
    }

    /// Normalised importance weights from log-likelihoods.
    pub fn importance_weights(&self) -> Vec<f64> {
        let log_scores = self.score_trajectories();
        let max_log = log_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log.is_infinite() {
            let n = log_scores.len();
            return vec![1.0 / n as f64; n];
        }
        let weights: Vec<f64> = log_scores.iter().map(|&l| (l - max_log).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    /// Index of the best-fitting trajectory (maximum likelihood).
    pub fn best_trajectory_index(&self) -> usize {
        let scores = self.score_trajectories();
        scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn n_trajectories(&self) -> usize {
        self.trajectories.len()
    }
}
