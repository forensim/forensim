"""Map evidence annotations to hypothesis likelihood weights."""

from __future__ import annotations

import math
from typing import Any

from forensim.annotate.manager import Annotation


def _tag_matches_events(tag: str, events: list[str]) -> bool:
    """Return True if an annotation tag semantically matches any event in the sequence."""
    tag_norm = tag.lower().replace(" ", "_").replace("-", "_")
    event_norms = {e.lower().replace(" ", "_").replace("-", "_") for e in events}

    # Direct substring match in either direction
    for event in event_norms:
        if tag_norm in event or event in tag_norm:
            return True

    # Common forensic synonyms
    synonyms: dict[str, list[str]] = {
        "blood_spatter": ["blood", "spatter", "spray"],
        "bullet_casing": ["casing", "shell", "brass"],
        "impact_point": ["impact", "hit", "strike"],
        "entry_wound": ["entry", "wound", "penetration"],
        "glass_fracture": ["glass", "fracture", "break", "shard"],
    }
    tag_tokens = set(tag_norm.split("_"))
    for event in event_norms:
        event_tokens = set(event.split("_"))
        for synonym_list in synonyms.values():
            if any(s in tag_tokens for s in synonym_list) and any(
                s in event_tokens for s in synonym_list
            ):
                return True
    return False


def _annotation_weight(annotation: Annotation) -> float:
    """Compute a positive weight for an annotation based on confidence and area."""
    return annotation.confidence


def apply_annotation_likelihoods(
    hypotheses: list[Any],
    annotations: list[Annotation],
    strength: float = 1.0,
) -> list[float]:
    """Compute log-likelihood offsets for hypotheses from annotations.

    Args:
        hypotheses: List of hypothesis objects with an ``events`` attribute.
        annotations: Annotated evidence regions.
        strength: Scaling factor for the likelihood effect.

    Returns:
        Log-likelihood offset per hypothesis (0 means no effect).
    """
    log_likelihoods: list[float] = []
    for hyp in hypotheses:
        events = getattr(hyp, "events", [])
        if not events:
            log_likelihoods.append(0.0)
            continue

        matched_weight = 0.0
        for annotation in annotations:
            if _tag_matches_events(annotation.tag, events):
                matched_weight += _annotation_weight(annotation)

        # Convert matched weight to a small log-likelihood bonus.
        # Using log1p keeps the effect bounded and interpretable.
        log_likelihoods.append(strength * math.log1p(matched_weight))
    return log_likelihoods
