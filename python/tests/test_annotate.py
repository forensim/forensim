"""Unit tests for the annotation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from forensim.annotate.manager import Annotation, AnnotationManager, Shape
from forensim.annotate.weights import apply_annotation_likelihoods
from forensim.infer.sequence import ScoredHypothesis, integrate_annotation_scores


@pytest.fixture
def sample_annotations() -> list[Annotation]:
    return [
        Annotation(
            id="a1",
            image_path="/evidence/img1.jpg",
            shape=Shape.RECT,
            coordinates=[[10, 10], [50, 50]],
            tag="blood_spatter",
            confidence=1.0,
        ),
        Annotation(
            id="a2",
            image_path="/evidence/img2.jpg",
            shape=Shape.POLYGON,
            coordinates=[[0, 0], [10, 0], [10, 10]],
            tag="bullet_casing",
            confidence=0.8,
        ),
    ]


def test_annotation_manager_save_and_load(sample_annotations: list[Annotation]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manager = AnnotationManager(Path(tmp))
        manager.save(sample_annotations)
        loaded = manager.load()
        assert len(loaded) == 2
        assert loaded[0].tag == "blood_spatter"
        assert loaded[1].shape == Shape.POLYGON


def test_annotation_manager_from_string_shape() -> None:
    a = Annotation(
        id="a1",
        image_path="img.jpg",
        shape="rect",
        coordinates=[[0, 0], [10, 10]],
        tag="impact_point",
    )
    assert a.shape == Shape.RECT


def test_apply_annotation_likelihoods_matches_events() -> None:
    class Hypothesis:
        def __init__(self, events: list[str]) -> None:
            self.events = events

    annotations = [
        Annotation(
            id="a1",
            image_path="img.jpg",
            shape="rect",
            coordinates=[[0, 0], [10, 10]],
            tag="blood_spatter",
            confidence=1.0,
        ),
    ]
    hypotheses = [Hypothesis(["blood_spatter", "fall"]), Hypothesis(["shot", "fall"])]
    log_likelihoods = apply_annotation_likelihoods(hypotheses, annotations, strength=1.0)
    assert log_likelihoods[0] > 0
    assert log_likelihoods[1] == 0.0


def test_integrate_annotation_scores_updates_ranking() -> None:
    hypotheses = [
        ScoredHypothesis(
            index=0,
            description="shot then blood",
            log_probability=-1.0,
            posterior=0.5,
            events=["shot", "blood_spatter"],
        ),
        ScoredHypothesis(
            index=1,
            description="shot then casing",
            log_probability=-0.5,
            posterior=0.5,
            events=["shot", "bullet_casing"],
        ),
    ]
    annotations = [
        Annotation(
            id="a1",
            image_path="img.jpg",
            shape="rect",
            coordinates=[[0, 0], [10, 10]],
            tag="bullet_casing",
            confidence=1.0,
        ),
    ]
    updated = integrate_annotation_scores(hypotheses, annotations, strength=2.0)
    # Casing hypothesis should now outrank blood hypothesis
    assert updated[0].description == "shot then casing"
    assert updated[0].posterior > updated[1].posterior


def test_integrate_annotation_scores_no_annotations() -> None:
    hypotheses = [
        ScoredHypothesis(
            index=0,
            description="H1",
            log_probability=-1.0,
            posterior=0.5,
            events=["shot"],
        ),
    ]
    updated = integrate_annotation_scores(hypotheses, [])
    assert updated[0].log_probability == -1.0
