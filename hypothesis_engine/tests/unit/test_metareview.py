"""Tests for MetaReviewAgent feedback incorporation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hypothesis_engine import ids
from hypothesis_engine.agents.base import AgentDeps
from hypothesis_engine.agents.metareview import MetaReviewAgent, _hypothesis_overview_context
from hypothesis_engine.llm.anthropic_client import AnthropicResponse
from hypothesis_engine.models import (
    CitedPaper,
    Hypothesis,
    ResearchPlan,
    Session,
    SystemFeedback,
    Task,
)
from hypothesis_engine.storage.repos import feedback as fb_repo
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import sessions as sess_repo


def _response(*, stop_reason: str, blocks: list[dict]) -> AnthropicResponse:
    content = []
    for block in blocks:
        content.append(
            SimpleNamespace(
                type=block.get("type", "text"),
                name=block.get("name", ""),
                input=block.get("input", {}),
                text=block.get("text", ""),
                id=block.get("id", ""),
            )
        )
    raw = SimpleNamespace(stop_reason=stop_reason, content=content)
    return AnthropicResponse(
        raw=raw,
        transcript_id="trn_metareview",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _session() -> Session:
    now = datetime.now(UTC)
    return Session(
        id="ses_metareview",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="Find mechanisms for defect formation.",
        research_plan=ResearchPlan(
            objective="Find mechanisms for defect formation.",
            preferences=["specificity"],
        ),
        config_snapshot={},
        budget_tokens=1_000_000,
        budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    )


def test_hypothesis_overview_context_preserves_grounding_report() -> None:
    text = (
        "# Hypothesis\n\n"
        + "study plan detail " * 600
        + "\n\n## Capability grounding\n\n"
        + "**Status.** validated\n\n"
        + "**Catalog revision.** `lab-v3`"
    )

    context = _hypothesis_overview_context(text, max_chars=1200)

    assert context.startswith("# Hypothesis")
    assert "[...middle omitted...]" in context
    assert "## Capability grounding" in context
    assert "**Status.** validated" in context
    assert "`lab-v3`" in context


@pytest.mark.asyncio
async def test_metareview_includes_human_feedback_without_reviews(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_directive",
            session_id=session.id,
            created_at=datetime.now(UTC),
            source="human",
            kind="directive",
            target_id=None,
            text="Prioritize hypotheses with immediate experimental validation.",
            active=True,
        ),
    )

    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "record_system_feedback",
                    "name": "record_system_feedback",
                    "input": {
                        "narrative": "Future proposals should emphasize direct validation.",
                        "common_weaknesses": [],
                        "common_strengths": [],
                        "suggested_focus_areas": ["direct validation"],
                    },
                }
            ],
        )
    )
    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="metareview",
        action="GenerateSystemFeedback",
        target_id=None,
        payload={"reason": "human_feedback", "feedback_id": "fb_directive"},
        priority=30,
        status="pending",
    )

    result = await agent.execute(task)

    assert result.kind == "system_feedback_generated"
    assert result.extra["n_reviews"] == 0
    assert result.extra["n_human_feedback"] == 1
    spec = llm.call.await_args.args[0]
    prompt = "\n".join(block.text for block in spec.user_blocks)
    assert "Active human feedback" in prompt
    assert "directive (global): Prioritize hypotheses" in prompt
    assert "Provided reviews for meta-analysis:\n(none yet)" in prompt
    latest = await fb_repo.latest_system_feedback(conn, session.id)
    assert latest is not None
    assert latest.text.startswith("Future proposals should emphasize direct validation.")


@pytest.mark.asyncio
async def test_metareview_empty_record_carries_forward_previous_feedback(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_previous_meta",
            session_id=session.id,
            created_at=datetime.now(UTC) - timedelta(minutes=5),
            source="meta_review",
            kind="system_feedback",
            target_id=None,
            text="Prior synthesis should remain available to future agents.",
            active=True,
        ),
    )
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_directive",
            session_id=session.id,
            created_at=datetime.now(UTC),
            source="human",
            kind="directive",
            target_id=None,
            text="Keep testing the metareview path.",
            active=True,
        ),
    )

    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "record_system_feedback",
                    "name": "record_system_feedback",
                    "input": {
                        "narrative": "",
                        "common_weaknesses": [],
                        "common_strengths": [],
                        "suggested_focus_areas": [],
                    },
                }
            ],
        )
    )
    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="metareview",
        action="GenerateSystemFeedback",
        target_id=None,
        payload={"reason": "periodic_tournament_threshold"},
        priority=95,
        status="pending",
    )

    result = await agent.execute(task)

    assert result.kind == "system_feedback_generated"
    latest = await fb_repo.latest_system_feedback(conn, session.id)
    assert latest is not None
    assert latest.id != "fb_previous_meta"
    assert latest.text == "Prior synthesis should remain available to future agents."


@pytest.mark.asyncio
async def test_final_overview_appends_stored_citations_without_prompting_llm(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(
        conn,
        Hypothesis(
            id="hyp_final_cited",
            session_id=session.id,
            created_at=datetime.now(UTC),
            created_by="generation",
            strategy="literature",
            title="Citation backed hypothesis",
            summary="A concise hypothesis summary without URLs.",
            full_text="full text",
            citations=[
                CitedPaper(
                    title="Final cited paper",
                    url="https://example.test/final-citation",
                    year=2024,
                )
            ],
            artifact_path=f"artifacts/{session.id}/hypotheses/hyp_final_cited.json",
            elo=1234.0,
            matches_played=3,
            state="in_tournament",
        ),
    )

    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="end_turn",
            blocks=[{"type": "text", "text": "# Executive summary\n\nModel-written overview."}],
        )
    )
    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="metareview",
        action="GenerateFinalResearchOverview",
        target_id=None,
        payload={},
        priority=1,
        status="pending",
    )

    result = await agent.execute(task)

    assert result.kind == "final_overview_generated"
    overview_path = tmp_cfg.data_dir / result.extra["overview_path"]
    text = overview_path.read_text(encoding="utf-8")
    assert "# Hypothesis citation sources" in text
    assert "Citation backed hypothesis" in text
    assert "Final cited paper" in text
    assert "https://example.test/final-citation" in text

    spec = llm.call.await_args.args[0]
    prompt_text = "\n".join(block.text for block in spec.user_blocks)
    assert "https://example.test/final-citation" not in prompt_text
    assert "**First catalog-grounded workflow.**" in prompt_text
    assert "Never describe a workflow as validated unless" in prompt_text


@pytest.mark.asyncio
async def test_metareview_recovers_unparseable_tool_arguments(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_recovery_directive",
            session_id=session.id,
            created_at=datetime.now(UTC),
            source="human",
            kind="directive",
            target_id=None,
            text="Preserve capability-grounded recommendations.",
            active=True,
        ),
    )

    malformed = _response(
        stop_reason="end_turn",
        blocks=[
            {
                "type": "tool_use",
                "name": "record_system_feedback",
                "input": {"_raw_arguments": "prose followed by a malformed tool call"},
            }
        ],
    )
    recovered = _response(
        stop_reason="tool_use",
        blocks=[
            {
                "type": "tool_use",
                "name": "record_system_feedback",
                "input": {
                    "narrative": "Use catalog-backed STEM and MD workflows.",
                    "common_weaknesses": ["Missing exact capability mapping"],
                    "common_strengths": ["Mechanistic specificity"],
                    "suggested_focus_areas": ["Validate capability availability"],
                },
            }
        ],
    )
    llm = MagicMock()
    llm.call = AsyncMock(side_effect=[malformed, recovered])
    agent = MetaReviewAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="metareview",
        action="GenerateSystemFeedback",
        payload={"reason": "human_feedback"},
    )

    result = await agent.execute(task)

    assert result.kind == "system_feedback_generated"
    assert llm.call.await_count == 2
    recovery_ctx = llm.call.await_args_list[1].args[1]
    assert recovery_ctx.mode == "system_recovery"
    latest = await fb_repo.latest_system_feedback(conn, session.id)
    assert latest is not None
    assert latest.text.startswith("Use catalog-backed STEM and MD workflows.")
