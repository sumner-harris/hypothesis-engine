# Modified from the original work.
"""Prompt rendering smoke."""

from __future__ import annotations

import pytest

from hypothesis_engine.llm import prompts


def test_all_templates_exist_on_disk() -> None:
    for key in prompts.TEMPLATES:
        p = prompts.template_path(key)
        assert p.exists(), f"missing template file for {key}: {p}"


def test_render_parse_goal() -> None:
    out = prompts.render(
        "parse_goal",
        goal="Investigate how X causes Y in mammalian cells",
        preferences_text="testable, specific",
    )
    assert "Investigate how X causes Y" in out
    assert "testable, specific" in out


def test_render_generation_literature() -> None:
    out = prompts.render(
        "generation.literature",
        goal="goal",
        preferences="prefs",
        articles_with_reasoning="(articles)",
    )
    assert "Goal: goal" in out
    assert "(articles)" in out
    assert "record_hypothesis" in out


def test_render_ranking_pairwise() -> None:
    out = prompts.render(
        "ranking.pairwise",
        goal="g",
        idea_attributes="novel, testable",
        hypothesis_1="H1 prose",
        hypothesis_1_id="H1",
        hypothesis_2="H2 prose",
        hypothesis_2_id="H2",
        review_1="R1",
        review_2="R2",
    )
    assert "better idea: <1 or 2>" in out
    assert "H1 prose" in out


def test_render_unknown_template_raises() -> None:
    with pytest.raises(KeyError):
        prompts.render("nonexistent.template")
