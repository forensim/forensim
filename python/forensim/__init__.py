"""
forensim — Forensic Scene Reconstruction & Probabilistic Event Analysis

Sub-packages:
    forensim.reconstruct   — Photogrammetry pipeline (COLMAP, gsplat, NuRec, USD export)
    forensim.simulate      — Physics simulation (PhysX via ovphysx)
    forensim.infer         — Probabilistic inference (Bayesian, HMM, Markov)
    forensim.api           — FastAPI REST sidecar

The Rust extension (forensim._core) provides high-performance implementations of:
    HiddenMarkovModel, MarkovChain, BayesianUpdater, MonteCarloEngine, summary_stats
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
