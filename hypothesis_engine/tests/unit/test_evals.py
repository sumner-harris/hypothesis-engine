# Modified from the original work.
"""Structural-eval tests (no judge calls)."""

from __future__ import annotations

import pytest

from hypothesis_engine.config import Config
from hypothesis_engine.evals.rubrics import GENERATION_RUBRIC, weighted_total
from hypothesis_engine.evals.runner import _check_structure, run_agent


def test_ranking_structural_check_requires_better_idea() -> None:
    bad = _check_structure("ranking", "no verdict here", {})
    assert any("better idea" in e for e in bad)
    good = _check_structure("ranking", "...\nbetter idea: 1", {})
    assert good == []


def test_generation_structural_check_finds_missing_sections() -> None:
    errs = _check_structure(
        "generation",
        "just a paragraph with no required sections",
        {},
    )
    assert any("mechanism" in e or "entit" in e for e in errs)


def test_reflection_structural_check_requires_citations() -> None:
    errs = _check_structure(
        "reflection",
        "review text without URLs",
        {"must_cite_at_least": 2},
    )
    assert any("URL" in e for e in errs)


def test_weighted_total_simple() -> None:
    scores = [
        {"name": "novelty", "score": 5, "rationale": ""},
        {"name": "specificity", "score": 3, "rationale": ""},
        {"name": "citation_grounding", "score": 4, "rationale": ""},
        {"name": "testability", "score": 4, "rationale": ""},
    ]
    w = weighted_total(GENERATION_RUBRIC, scores)
    # (5+3+4+4) / (5*4) = 0.8
    assert w == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_offline_run_uses_no_judge_call() -> None:
    cfg = Config()
    # ranking.jsonl is bundled; offline=True must not call the API.
    result = await run_agent(cfg, "ranking", offline=True)
    assert result["n_fixtures"] >= 1
    # offline → no `mean_weighted`
    assert result["mean_weighted"] is None
