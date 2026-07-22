# Modified from the original work.
"""Tests for the ranking verdict parser and mode-selection logic."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from hypothesis_engine.agents.ranking import (
    RankingAgent,
    _pair_key,
    _parse_better_idea,
    _prompt_match_fields,
    _stop_reason,
)
from hypothesis_engine.config import Config
from hypothesis_engine.models import Hypothesis

# ----------------------------- verdict parser ----------------------------- #

def test_parse_better_idea_basic() -> None:
    assert _parse_better_idea("blah\nbetter idea: 1") == 1
    assert _parse_better_idea("blah\nbetter idea: 2") == 2


def test_parse_better_idea_trailing_marker_wins() -> None:
    text = "An earlier mention: better idea: 1\n\nFinal verdict.\nbetter idea: 2"
    assert _parse_better_idea(text) == 2


def test_parse_better_idea_handles_case_and_punctuation() -> None:
    assert _parse_better_idea("...\nBetter Idea: 2.") == 2
    assert _parse_better_idea("...\n**better idea**: 1") == 1


def test_parse_better_idea_returns_none_when_missing() -> None:
    assert _parse_better_idea("no verdict here") is None
    assert _parse_better_idea("") is None


def test_parse_better_idea_handles_qualifier_words() -> None:
    """Regression: the prior 'in tail.split()[0:1]' check rejected these."""
    assert _parse_better_idea("better idea: option 1") == 1
    assert _parse_better_idea("better idea: hypothesis 2") == 2
    assert _parse_better_idea("better idea: hyp 1") == 1


def test_parse_better_idea_word_boundary_excludes_12() -> None:
    """'better idea: 12 because...' must NOT be read as '1'."""
    assert _parse_better_idea("better idea: 12 because of context") is None


# ----------------------------- mode selection ----------------------------- #

def _h(*, elo: float, matches: int, hid: str = "hyp_x") -> Hypothesis:
    return Hypothesis(
        id=hid, session_id="ses", created_at=datetime.now(UTC),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/ses/hypotheses/{hid}.json",
        elo=elo, matches_played=matches, state="in_tournament",
    )


def _agent() -> RankingAgent:
    deps = MagicMock()
    deps.cfg = Config()
    return RankingAgent(deps)


def test_prompt_layout_uses_storage_order_and_records_metadata() -> None:
    agent = _agent()
    agent.deps.cfg.ranking.prompt_hypothesis_max_chars = 80
    agent.deps.cfg.ranking.prompt_review_max_chars = 40
    agent.deps.cfg.ranking.prompt_side_max_chars = 120
    a = _h(hid="a", elo=1200, matches=2)
    b = _h(hid="b", elo=1200, matches=2)
    a.full_text = "A" * 500
    b.full_text = "B" * 500

    layout = agent._build_prompt_layout("ses", "task", a, b, "review a" * 50, "review b" * 50)

    fields = _prompt_match_fields(layout, winner="b")
    assert fields["prompt1_hyp_id"] == "a"
    assert fields["prompt2_hyp_id"] == "b"
    assert fields["prompt1_side"] == "a"
    assert fields["prompt2_side"] == "b"
    assert fields["winner_prompt_position"] == 2
    assert fields["prompt1_chars"] <= 120
    assert fields["prompt2_chars"] <= 120


def test_prompt_layout_negative_limits_keep_full_text() -> None:
    agent = _agent()
    agent.deps.cfg.ranking.prompt_hypothesis_max_chars = -1
    agent.deps.cfg.ranking.prompt_review_max_chars = -1
    agent.deps.cfg.ranking.prompt_side_max_chars = -1
    a = _h(hid="a", elo=1200, matches=2)
    b = _h(hid="b", elo=1200, matches=2)
    a.full_text = "A" * 500
    b.full_text = "B" * 500
    review_a = "review a" * 50
    review_b = "review b" * 50

    layout = agent._build_prompt_layout("ses", "task", a, b, review_a, review_b)

    assert "A" * 500 in layout.prompt1.hypothesis_text
    assert "B" * 500 in layout.prompt2.hypothesis_text
    assert layout.prompt1.review_text == review_a
    assert layout.prompt2.review_text == review_b
    assert "truncated to balanced ranking prompt budget" not in layout.prompt1.hypothesis_text
    assert "truncated to balanced ranking prompt budget" not in layout.prompt2.hypothesis_text


def test_prompt_layout_zero_limits_clip_to_empty_text() -> None:
    agent = _agent()
    agent.deps.cfg.ranking.prompt_hypothesis_max_chars = 0
    agent.deps.cfg.ranking.prompt_review_max_chars = 0
    agent.deps.cfg.ranking.prompt_side_max_chars = 0
    a = _h(hid="a", elo=1200, matches=2)
    b = _h(hid="b", elo=1200, matches=2)
    a.full_text = "A" * 500
    b.full_text = "B" * 500

    layout = agent._build_prompt_layout("ses", "task", a, b, "review a", "review b")

    assert layout.prompt1.hypothesis_text == ""
    assert layout.prompt1.review_text == ""
    assert layout.prompt2.hypothesis_text == ""
    assert layout.prompt2.review_text == ""


def test_mode_debate_when_either_player_has_few_matches() -> None:
    a = _h(hid="a", elo=1500, matches=0)
    b = _h(hid="b", elo=1500, matches=10)
    assert _agent()._select_mode(a, b) == "debate"


def test_mode_debate_when_elo_gap_is_small() -> None:
    a = _h(hid="a", elo=1500, matches=5)
    b = _h(hid="b", elo=1520, matches=5)
    assert _agent()._select_mode(a, b) == "debate"


def test_mode_pairwise_when_warm_and_large_gap() -> None:
    a = _h(hid="a", elo=1500, matches=10)
    b = _h(hid="b", elo=1300, matches=10)
    assert _agent()._select_mode(a, b) == "pairwise"


# ----------------------------- nearest-Elo helper ----------------------------- #

def test_nearest_elo_picks_closest() -> None:
    target = _h(hid="t", elo=1300, matches=0)
    pool = [
        _h(hid="a", elo=1000, matches=5),
        _h(hid="b", elo=1310, matches=5),
        _h(hid="c", elo=1500, matches=5),
    ]
    nearest = _agent()._nearest_elo(target, pool)
    assert nearest is not None and nearest.id == "b"


def test_nearest_elo_empty_pool() -> None:
    target = _h(hid="t", elo=1300, matches=0)
    assert _agent()._nearest_elo(target, []) is None


def test_nearest_elo_prefers_less_played_comparable_opponent() -> None:
    target = _h(hid="t", elo=1200, matches=0)
    pool = [
        _h(hid="overused_anchor", elo=1200, matches=24),
        _h(hid="balanced_nearby", elo=1216, matches=1),
        _h(hid="too_far", elo=1300, matches=0),
    ]

    nearest = _agent()._nearest_elo(target, pool)

    assert nearest is not None and nearest.id == "balanced_nearby"


def test_nearest_elo_keeps_large_elo_gaps_outside_balance_window() -> None:
    target = _h(hid="t", elo=1200, matches=0)
    pool = [
        _h(hid="closest_anchor", elo=1200, matches=24),
        _h(hid="underplayed_but_far", elo=1260, matches=0),
    ]

    nearest = _agent()._nearest_elo(target, pool)

    assert nearest is not None and nearest.id == "closest_anchor"


def test_nearest_elo_prefers_more_similar_when_match_counts_tie(monkeypatch) -> None:
    agent = _agent()
    target = _h(hid="t", elo=1200, matches=0)
    pool = [
        _h(hid="exact_but_less_similar", elo=1200, matches=2),
        _h(hid="nearby_and_more_similar", elo=1216, matches=2),
    ]

    def _fake_similarity(_store, _target, candidate):
        return {
            "exact_but_less_similar": 0.5,
            "nearby_and_more_similar": 0.95,
        }[candidate.id]

    monkeypatch.setattr(agent, "_similarity", _fake_similarity)

    nearest = agent._nearest_elo(target, pool, store=object())

    assert nearest is not None and nearest.id == "nearby_and_more_similar"


@pytest.mark.asyncio
async def test_select_pair_avoids_reserved_focus_pair(monkeypatch) -> None:
    agent = _agent()

    async def _no_store(_session_id: str):
        return None

    monkeypatch.setattr(agent, "_load_store", _no_store)
    candidates = [
        _h(hid="target", elo=1200, matches=0),
        _h(hid="closest_reserved", elo=1200, matches=4),
        _h(hid="comparable_fallback", elo=1216, matches=4),
    ]

    first, second, _similarity = await agent._select_pair(
        "ses",
        candidates,
        focus_id="target",
        blocked_pair_keys={_pair_key("target", "closest_reserved")},
    )

    assert first.id == "target"
    assert second.id == "comparable_fallback"


@pytest.mark.asyncio
async def test_select_pair_prioritizes_warmup_before_evolution_gate(monkeypatch) -> None:
    agent = _agent()
    agent.deps.cfg.evolution.min_mature = 5

    async def _no_store(_session_id: str):
        return None

    monkeypatch.setattr(agent, "_load_store", _no_store)
    candidates = [
        _h(hid="warm_a", elo=1250, matches=12),
        _h(hid="warm_b", elo=1160, matches=8),
        _h(hid="nearly_mature", elo=1240, matches=2),
        _h(hid="less_warm", elo=1400, matches=1),
        _h(hid="new", elo=1100, matches=0),
    ]

    first, second, _similarity = await agent._select_pair(
        "ses", candidates, focus_id=None
    )

    assert first.id == "nearly_mature"
    assert second.id == "warm_a"


def test_ranking_output_caps_are_configurable() -> None:
    cfg = Config()
    assert cfg.ranking.pairwise_max_output_tokens == 8192
    assert cfg.ranking.debate_max_output_tokens == 12288
    assert cfg.ranking.verdict_retry_max_output_tokens == 1024

    cfg.ranking.pairwise_max_output_tokens = 4096
    cfg.ranking.debate_max_output_tokens = 16384
    cfg.ranking.verdict_retry_max_output_tokens = 2048
    assert cfg.ranking.pairwise_max_output_tokens == 4096
    assert cfg.ranking.debate_max_output_tokens == 16384
    assert cfg.ranking.verdict_retry_max_output_tokens == 2048


def test_stop_reason_helper_reads_provider_stop_reason() -> None:
    response = MagicMock()
    response.raw.stop_reason = "max_tokens"
    assert _stop_reason(response) == "max_tokens"
