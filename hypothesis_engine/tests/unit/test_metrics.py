# Modified from the original work.
"""Tests for the obs/metrics aggregations."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hypothesis_engine.models import Hypothesis, ResearchPlan, Session, Transcript
from hypothesis_engine.obs.metrics import session_metrics, to_dict
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import sessions as sess_repo
from hypothesis_engine.storage.repos import transcripts as tx_repo


async def _seed(conn, session_id: str = "ses_m") -> None:
    now = datetime.now(UTC)
    await sess_repo.insert(conn, Session(
        id=session_id, created_at=now, updated_at=now, status="running",
        research_goal="g", research_plan=ResearchPlan(objective="o"),
        config_snapshot={}, budget_tokens=1_000_000, budget_usd=10.0,
    ))
    await hyp_repo.insert(conn, Hypothesis(
        id="hyp_x", session_id=session_id, created_at=now,
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/{session_id}/hypotheses/hyp_x.json",
        state="in_tournament", elo=1234.0, matches_played=3,
    ))
    await tx_repo.insert(conn, Transcript(
        id="trn_1", session_id=session_id, task_id=None,
        agent="generation", action="x", model="claude-opus-4-7",
        input_tokens=1000, output_tokens=200, cache_read=500, cache_write=0,
        cost_usd=0.12, started_at=now, finished_at=now,
        artifact_path=f"artifacts/{session_id}/transcripts/generation/trn_1.json",
    ))


@pytest.mark.asyncio
async def test_session_metrics_aggregates(conn) -> None:
    await _seed(conn)
    m = await session_metrics(conn, "ses_m")
    assert m.n_calls == 1
    assert m.input_tokens == 1000
    assert m.n_hypotheses == 1
    assert m.n_in_tournament == 1
    assert m.cost_usd == pytest.approx(0.12)
    # cache_hit_ratio = 500 / (500 + 0 + 1000) = 1/3
    assert m.cache_hit_ratio == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_session_metrics_to_dict_roundtrips(conn) -> None:
    await _seed(conn, session_id="ses_m2")
    m = await session_metrics(conn, "ses_m2")
    d = to_dict(m)
    for key in (
        "n_calls", "input_tokens", "n_hypotheses",
        "n_in_tournament", "cost_usd", "cache_hit_ratio",
    ):
        assert key in d
