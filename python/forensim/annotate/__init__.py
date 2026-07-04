"""Evidence annotation utilities: ROI drawing and tag-based likelihood weighting."""

from __future__ import annotations

from .manager import Annotation, AnnotationManager, Shape
from .weights import apply_annotation_likelihoods

__all__ = [
    "Annotation",
    "AnnotationManager",
    "Shape",
    "apply_annotation_likelihoods",
]
