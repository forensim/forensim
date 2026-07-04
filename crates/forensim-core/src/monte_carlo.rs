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
    /// `trajectory` is a list of [x, y, z] triples.
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
                    log_p += -d2 / (2.0 * sigma * sigma);
                }
                log_p
            })
            .collect()
    }

    /// Score a single trajectory at a specific timestep against the evidence.
    ///
    /// Returns NEG_INFINITY if indices are out of bounds.
    pub fn score_trajectory_at_time(&self, traj_idx: usize, time_step: usize) -> f64 {
        let traj = match self.trajectories.get(traj_idx) {
            Some(t) => t,
            None => return f64::NEG_INFINITY,
        };
        let pos = match traj.get(time_step) {
            Some(p) => p,
            None => return f64::NEG_INFINITY,
        };
        let sigma = self.position_sigma;
        let mut log_p = 0.0f64;
        for ev in &self.evidence_positions {
            let d2 = (0..3).map(|k| (pos[k] - ev[k]).powi(2)).sum::<f64>();
            log_p += -d2 / (2.0 * sigma * sigma);
        }
        log_p
    }

    /// Normalised importance weights from log-likelihoods.
    pub fn importance_weights(&self) -> Vec<f64> {
        let log_scores = self.score_trajectories();
        let max_log = log_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_log.is_infinite() {
            let n = log_scores.len();
            if n == 0 {
                return vec![];
            }
            return vec![1.0 / n as f64; n];
        }
        let weights: Vec<f64> = log_scores.iter().map(|&l| (l - max_log).exp()).collect();
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| w / total).collect()
    }

    /// Effective Sample Size: ESS = 1 / Σ(w_i²).
    ///
    /// For uniform weights, ESS = N (full efficiency).
    /// Low ESS signals weight degeneracy (few trajectories dominate).
    pub fn effective_sample_size(&self) -> f64 {
        let weights = self.importance_weights();
        if weights.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
        if sum_sq <= 0.0 { 0.0 } else { 1.0 / sum_sq }
    }

    /// Importance-weighted mean of final positions across all trajectories.
    ///
    /// Returns [0.0, 0.0, 0.0] if no trajectories are stored.
    pub fn weighted_mean_position(&self) -> [f64; 3] {
        if self.trajectories.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        let weights = self.importance_weights();
        let mut mean = [0.0f64; 3];
        for (traj, &w) in self.trajectories.iter().zip(weights.iter()) {
            if let Some(last) = traj.last() {
                for k in 0..3 {
                    mean[k] += w * last[k];
                }
            }
        }
        mean
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn engine_at_origin() -> MonteCarloEngine {
        MonteCarloEngine::new(vec![[0.0, 0.0, 0.0]], 1.0)
    }

    #[test]
    fn test_score_empty() {
        let e = engine_at_origin();
        let scores: Vec<f64> = e.score_trajectories();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_importance_weights_sum_to_one() {
        let mut e = engine_at_origin();
        e.add_trajectory(vec![[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        e.add_trajectory(vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        e.add_trajectory(vec![[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]);
        let w = e.importance_weights();
        assert_relative_eq!(w.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_best_trajectory_index_closest_wins() {
        let mut e = engine_at_origin();
        e.add_trajectory(vec![[10.0, 0.0, 0.0]]); // far
        e.add_trajectory(vec![[0.1, 0.0, 0.0]]); // close
        assert_eq!(e.best_trajectory_index(), 1);
    }

    #[test]
    fn test_effective_sample_size_uniform() {
        let mut e = engine_at_origin();
        // Four identical trajectories → uniform weights → ESS = 4
        for _ in 0..4 {
            e.add_trajectory(vec![[0.0, 0.0, 0.0]]);
        }
        let ess = e.effective_sample_size();
        assert_relative_eq!(ess, 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_effective_sample_size_empty() {
        assert_eq!(engine_at_origin().effective_sample_size(), 0.0);
    }

    #[test]
    fn test_weighted_mean_no_trajectories() {
        assert_eq!(engine_at_origin().weighted_mean_position(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_weighted_mean_single() {
        let mut e = engine_at_origin();
        e.add_trajectory(vec![[3.0, 4.0, 5.0]]);
        let mean = e.weighted_mean_position();
        assert_relative_eq!(mean[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(mean[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_score_trajectory_at_time_oob() {
        let e = engine_at_origin();
        assert_eq!(e.score_trajectory_at_time(0, 0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_score_trajectory_at_time_valid() {
        let mut e = engine_at_origin();
        e.add_trajectory(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let s0 = e.score_trajectory_at_time(0, 0); // at origin — high score
        let s1 = e.score_trajectory_at_time(0, 1); // 1m away — lower score
        assert!(s0 > s1);
    }

    #[test]
    fn test_clear_trajectories() {
        let mut e = engine_at_origin();
        e.add_trajectory(vec![[0.0, 0.0, 0.0]]);
        e.clear_trajectories();
        assert_eq!(e.n_trajectories(), 0);
    }
}
