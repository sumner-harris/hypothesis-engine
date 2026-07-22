# Modified from the original work.
"""Tests for the safety classifier's action mapping (no API calls)."""

from __future__ import annotations

from hypothesis_engine.config import Config
from hypothesis_engine.safety.classifier import ClassifierResult


def test_benign_is_allowed() -> None:
    r = ClassifierResult(categories=["none"], confidence=1.0, rationale="ok")
    assert r.is_benign
    assert r.action(Config()) == "allow"


def test_block_categories_block() -> None:
    cfg = Config()
    r = ClassifierResult(categories=["cbrn"], confidence=0.99, rationale="x")
    assert not r.is_benign
    assert r.action(cfg) == "block"


def test_warn_high_confidence_quarantines() -> None:
    cfg = Config()
    r = ClassifierResult(categories=["dual_use_bio"], confidence=0.8, rationale="x")
    assert r.action(cfg) == "quarantine"


def test_warn_low_confidence_warns() -> None:
    cfg = Config()
    r = ClassifierResult(categories=["dual_use_bio"], confidence=0.4, rationale="x")
    assert r.action(cfg) == "warn"


def test_unflagged_other_category_allows() -> None:
    cfg = Config()
    r = ClassifierResult(categories=["unknown_label"], confidence=0.5, rationale="x")
    assert r.action(cfg) == "allow"
