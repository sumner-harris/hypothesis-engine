# Modified from the original work.
"""Tests for the pre-flight cost estimator."""

from __future__ import annotations

from hypothesis_engine.config import Config
from hypothesis_engine.llm.estimator import estimate


def test_estimate_emits_warning_when_budget_too_low() -> None:
    cfg = Config()
    cfg.run.budget_usd = 1.0
    est = estimate(cfg)
    assert est.total_usd > cfg.run.budget_usd
    assert est.warning is not None and "exceeds" in est.warning


def test_estimate_no_warning_when_budget_generous() -> None:
    cfg = Config()
    cfg.run.budget_usd = 9999.0
    est = estimate(cfg)
    assert est.warning is None


def test_estimate_rows_include_all_phases() -> None:
    cfg = Config()
    est = estimate(cfg)
    labels = {r.label for r in est.rows}
    assert {
        "parse_goal", "generation", "reflection.full",
        "ranking_pairwise", "metareview.final",
    } <= labels


def test_estimate_scales_with_max_ideas() -> None:
    cfg = Config()
    small = estimate(cfg, max_ideas=10, max_matches_per_idea=4)
    big = estimate(cfg, max_ideas=100, max_matches_per_idea=12)
    assert big.total_usd > small.total_usd * 5
