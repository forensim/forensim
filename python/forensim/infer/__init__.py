"""
forensim.infer — Probabilistic inference for forensic event reconstruction.

Modules:
    bayesian   — PyMC Bayesian models and posterior sampling
    sequence   — Event sequence likelihood computation (wraps forensim._core)
    evidence   — Evidence scoring and likelihood ratio computation
    sensitivity — Leave-one-out sensitivity analysis for evidence sources
"""

from forensim.infer.sensitivity import (
    EvidenceSource,
    SensitivityResult,
    compute_annotation_sensitivity,
    compute_sensitivity,
)

__all__ = [
    "EvidenceSource",
    "SensitivityResult",
    "compute_annotation_sensitivity",
    "compute_sensitivity",
]
