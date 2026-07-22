"""Tests for EvolutionAgent strategy scheduling and recovery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import hypothesis_engine.agents.evolution as evolution_mod
from hypothesis_engine.agents.base import AgentDeps
from hypothesis_engine.agents.evolution import EvolutionAgent
from hypothesis_engine.llm.anthropic_client import AnthropicResponse
from hypothesis_engine.llm.tool_loop import ToolLoopResult
from hypothesis_engine.models import Hypothesis, ResearchPlan, Session, Task
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import sessions as sess_repo
from hypothesis_engine.vectors.store import FaissStore


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
        transcript_id="trn_evolution",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _study_plan(strategy: str) -> list[dict]:
    return [
        {
            "component_id": "evolved_validation",
            "component_label": "Evolved validation",
            "role": "validation",
            "objective": f"Test the {strategy} evolved mechanism against parent hypotheses.",
            "methods": ["Compare simulated or experimental defect signatures across strategies."],
            "variables": ["ion energy", "fluence", "substrate"],
            "outputs": ["defect contrast", "vacancy distribution"],
            "quantitative_targets": ["distinct response relative to parent hypotheses"],
            "controls_or_comparators": ["parent hypothesis conditions"],
            "failure_criteria": ["response is indistinguishable from parent hypotheses"],
        }
    ]


def _record(strategy: str) -> dict:
    details = {
        "combine": (
            "coupled substrate recoil and chalcogen momentum focusing",
            "aligned divacancy clusters after grazing incidence exposure",
        ),
        "simplify": (
            "single-collision chalcogen threshold lowering",
            "isolated sulfur vacancies scaling with transferred recoil energy",
        ),
        "out_of_box": (
            "transient exciton-polaron bond weakening before impact",
            "defects following optical excitation density rather than ion fluence alone",
        ),
    }.get(strategy, ("generic ion damage", "generic defect contrast"))
    mechanism, outcome = details
    return {
        "title": f"{strategy} evolved hypothesis",
        "statement": f"{strategy} proposes {mechanism} in implanted monolayer TMDs.",
        "mechanism": f"The evolved mechanism centers on {mechanism}, producing a distinct implantation response.",
        "entities": [strategy, "low energy ion", "monolayer TMD"],
        "anticipated_outcomes": outcome,
        "novelty_argument": f"The {strategy} strategy changes the causal bottleneck relative to the parent ideas.",
        "study_plan": _study_plan(strategy),
        "citations": [],
    }


def _session() -> Session:
    now = datetime.now(UTC)
    return Session(
        id="ses_evolution",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="Find mechanisms for defect formation.",
        research_plan=ResearchPlan(objective="Find mechanisms for defect formation."),
        config_snapshot={},
        budget_tokens=1_000_000,
        budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    )


def _hypothesis(hid: str, session_id: str, *, elo: float) -> Hypothesis:
    return Hypothesis(
        id=hid,
        session_id=session_id,
        created_at=datetime.now(UTC),
        created_by="generation",
        strategy="literature",
        title=hid,
        summary="summary",
        full_text=f"# {hid}\n\nA parent hypothesis.",
        artifact_path=f"artifacts/{hid}.md",
        elo=elo,
        matches_played=3,
        state="in_tournament",
    )


async def _seed_session(conn) -> Session:
    session = _session()
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, _hypothesis("hyp_a", session.id, elo=1210.0))
    await hyp_repo.insert(conn, _hypothesis("hyp_b", session.id, elo=1200.0))
    return session


async def _seed_clustered_session(conn, tmp_cfg) -> tuple[Session, list[Hypothesis]]:
    tmp_cfg.embeddings.provider = "hash"
    tmp_cfg.embeddings.dim = 4
    session = _session()
    await sess_repo.insert(conn, session)

    specs = [
        ("hyp_a1", 1400.0, [1.0, 0.0, 0.0, 0.0]),
        ("hyp_a2", 1390.0, [1.0, 0.0, 0.0, 0.0]),
        ("hyp_a3", 1380.0, [1.0, 0.0, 0.0, 0.0]),
        ("hyp_b1", 1250.0, [0.0, 1.0, 0.0, 0.0]),
        ("hyp_b2", 1240.0, [0.0, 1.0, 0.0, 0.0]),
        ("hyp_b3", 1230.0, [0.0, 1.0, 0.0, 0.0]),
    ]
    store = FaissStore(tmp_cfg, session.id, dim=4)
    hypotheses: list[Hypothesis] = []
    for hid, elo, vec in specs:
        h = _hypothesis(hid, session.id, elo=elo)
        hypotheses.append(h)
        await hyp_repo.insert(conn, h)
        await store.add_and_save(h.id, np.asarray(vec, dtype="float32"))
    return session, hypotheses


@pytest.mark.asyncio
async def test_evolution_parent_selection_balances_embedding_clusters(conn, tmp_cfg) -> None:
    session, _hypotheses = await _seed_clustered_session(conn, tmp_cfg)
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )
    candidates = await hyp_repo.tournament_candidates(conn, session.id)

    selection = await agent._select_cluster_balanced_top(
        session.id,
        candidates,
        top_k=3,
    )

    selected_ids = [h.id for h in selection.hypotheses]
    assert selected_ids == ["hyp_a1", "hyp_b1", "hyp_a2"]
    assert selection.extra["mode"] == "cluster_balanced"
    assert selection.extra["cluster_count"] == 2
    assert {row["representative_id"] for row in selection.extra["cluster_representatives"]} == {
        "hyp_a1",
        "hyp_b1",
    }


@pytest.mark.asyncio
async def test_evolution_parent_selection_clusters_when_all_candidates_fit(conn, tmp_cfg) -> None:
    session, _hypotheses = await _seed_clustered_session(conn, tmp_cfg)
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )
    candidates = await hyp_repo.tournament_candidates(conn, session.id)

    selection = await agent._select_cluster_balanced_top(
        session.id,
        candidates,
        top_k=6,
    )

    assert selection.extra["mode"] == "cluster_balanced"
    assert selection.extra["cluster_count"] == 2
    assert [h.id for h in selection.hypotheses] == [
        "hyp_a1",
        "hyp_b1",
        "hyp_a2",
        "hyp_a3",
        "hyp_b2",
        "hyp_b3",
    ]


@pytest.mark.asyncio
async def test_evolution_parent_selection_focus_preserves_cluster_representatives(
    conn, tmp_cfg
) -> None:
    session, hypotheses = await _seed_clustered_session(conn, tmp_cfg)
    focus = next(h for h in hypotheses if h.id == "hyp_a3")
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )
    candidates = await hyp_repo.tournament_candidates(conn, session.id)

    selection = await agent._select_cluster_balanced_top(
        session.id,
        candidates,
        top_k=3,
        focus=focus,
    )

    assert [h.id for h in selection.hypotheses] == ["hyp_a3", "hyp_a1", "hyp_b1"]
    assert {row["representative_id"] for row in selection.extra["cluster_representatives"]} == {
        "hyp_a1",
        "hyp_b1",
    }


@pytest.mark.asyncio
async def test_evolution_execute_uses_cluster_balanced_parent_set(conn, tmp_cfg) -> None:
    session, _hypotheses = await _seed_clustered_session(conn, tmp_cfg)
    captured_parent_ids: list[list[str]] = []
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=tools,
        )
    )

    async def fake_run_record(_session, attempt, *, task_id):
        captured_parent_ids.append(attempt.parent_ids)
        return evolution_mod.EvolutionRecordResult(
            record=_record(attempt.strategy),
            seen_urls=set(),
            recovered=False,
        )

    agent._run_record = fake_run_record
    agent._persist_detail = AsyncMock(return_value=evolution_mod.PersistResult("hyp_evolved", True))
    task = Task(
        id="tsk_clustered_evolution",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="evolution",
        action="EvolveTopHypotheses",
        payload={"top_k": 3, "strategies": ["out_of_box"]},
    )

    result = await agent.execute(task)

    assert captured_parent_ids == [["hyp_a1", "hyp_b1", "hyp_a2"]]
    assert result.extra["parent_selection"]["mode"] == "cluster_balanced"
    assert result.extra["parent_selection"]["top_ids"] == ["hyp_a1", "hyp_b1", "hyp_a2"]


@pytest.mark.asyncio
async def test_evolution_persists_successful_strategy_before_later_strategy_fails(
    monkeypatch, conn, tmp_cfg
) -> None:
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    session = await _seed_session(conn)
    modes: list[str] = []
    seen_url_gates: list[int] = []
    persisted_before_failure = False

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        nonlocal persisted_before_failure
        ctx = kwargs["ctx"]
        modes.append(ctx.mode)
        seen_url_gates.append(kwargs["terminal_min_seen_urls"])
        if ctx.mode == "combine":
            return ToolLoopResult(
                response=_response(
                    stop_reason="tool_use",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": f"record_{ctx.mode}",
                            "name": "record_hypothesis",
                            "input": _record(ctx.mode),
                        }
                    ],
                ),
                iterations=1,
                tool_calls=[],
                seen_urls=set(),
            )

        hyps = await hyp_repo.list_for_session(conn, session.id)
        persisted_before_failure = any(
            h.created_by == "evolution" and h.strategy == "combine" for h in hyps
        )
        raise TimeoutError("strategy timed out")

    monkeypatch.setattr(evolution_mod, "run_tool_loop", fake_tool_loop)
    llm = MagicMock()
    llm.call = AsyncMock()
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = EvolutionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_evolution",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="evolution",
        action="EvolveTopHypotheses",
        payload={"top_k": 2, "strategies": ["combine", "simplify"]},
    )

    result = await agent.execute(task)

    assert result.kind == "hypothesis_created"
    assert modes == ["combine", "simplify"]
    assert seen_url_gates == [0, 0]
    assert persisted_before_failure is True
    assert len(result.hypothesis_ids) == 1
    assert result.extra["strategies_succeeded"] == ["combine"]
    assert result.extra["failed_strategies"] == [
        {"strategy": "simplify", "error": "strategy timed out"}
    ]


@pytest.mark.asyncio
async def test_evolution_recovers_missing_record_hypothesis(monkeypatch, conn, tmp_cfg) -> None:
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    session = await _seed_session(conn)

    async def fake_tool_loop(*_args, **_kwargs) -> ToolLoopResult:
        return ToolLoopResult(
            response=_response(
                stop_reason="end_turn",
                blocks=[
                    {"type": "text", "text": "Here is a useful evolved idea, but no tool call."}
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(evolution_mod, "run_tool_loop", fake_tool_loop)
    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "recover_hypothesis",
                    "name": "record_hypothesis",
                    "input": _record("combine"),
                }
            ],
        )
    )
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = EvolutionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_evolution",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="evolution",
        action="EvolveTopHypotheses",
        payload={"top_k": 2, "strategies": ["combine"]},
    )

    result = await agent.execute(task)

    assert result.kind == "hypothesis_created"
    assert len(result.hypothesis_ids) == 1
    assert result.extra["recovered_record_hypotheses"] == [
        {
            "strategy": "combine",
            "attempts": 1,
            "max_output_tokens": tmp_cfg.evolution.hypothesis_recovery_max_output_tokens,
            "reason": "missing record_hypothesis tool call",
        }
    ]
    spec = llm.call.await_args.args[0]
    assert spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    assert spec.max_output_tokens == tmp_cfg.evolution.hypothesis_recovery_max_output_tokens
    ctx = llm.call.await_args.args[1]
    assert ctx.task_id == task.id


@pytest.mark.asyncio
async def test_evolution_persists_missing_title_without_recovery(
    monkeypatch, conn, tmp_cfg
) -> None:
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    session = await _seed_session(conn)
    record = _record("combine")
    record.pop("title")

    async def fake_tool_loop(*_args, **_kwargs) -> ToolLoopResult:
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "hypothesis",
                        "name": "record_hypothesis",
                        "input": record,
                    }
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(evolution_mod, "run_tool_loop", fake_tool_loop)
    llm = MagicMock()
    llm.call = AsyncMock()
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = EvolutionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_evolution_missing_title",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="evolution",
        action="EvolveTopHypotheses",
        payload={"top_k": 2, "strategies": ["combine"]},
    )

    result = await agent.execute(task)

    assert result.kind == "hypothesis_created"
    assert len(result.hypothesis_ids) == 1
    assert result.extra["recovered_record_hypotheses"] == []
    llm.call.assert_not_called()
    persisted = await hyp_repo.fetch(conn, result.hypothesis_ids[0])
    assert persisted is not None
    assert persisted.title.startswith("combine proposes coupled substrate recoil")


@pytest.mark.asyncio
async def test_evolution_persist_respects_max_ideas_cap(conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    tmp_cfg.run.max_ideas = 2
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )

    _hid, was_new = await agent._persist(session.id, _record("combine"), strategy="combine")

    assert was_new is False
    assert await hyp_repo.count_for_session(conn, session.id) == 2


@pytest.mark.asyncio
async def test_evolution_replaces_dedup_duplicate(monkeypatch, conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    tmp_cfg.evolution.dedup_replacement_attempts = 2
    await hyp_repo.insert(
        conn,
        Hypothesis(
            id="hyp_duplicate",
            session_id=session.id,
            created_at=datetime.now(UTC),
            created_by="generation",
            strategy="literature",
            title="Duplicate evolved hypothesis",
            summary="A duplicate evolved mechanism to avoid.",
            full_text="duplicate",
            artifact_path="artifacts/duplicate.md",
            state="draft",
        ),
    )
    agent = EvolutionAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )
    agent._run_record = AsyncMock(
        return_value=evolution_mod.EvolutionRecordResult(
            record=_record("combine"),
            seen_urls=set(),
            recovered=False,
        )
    )
    agent._persist_detail = AsyncMock(
        side_effect=[
            evolution_mod.PersistResult(
                "hyp_duplicate", False, reason="dedup_duplicate", duplicate_id="hyp_duplicate"
            ),
            evolution_mod.PersistResult("hyp_replacement", True),
        ]
    )

    async def replacement(**kwargs):
        assert kwargs["duplicate"].title == "Duplicate evolved hypothesis"
        assert kwargs["attempt_n"] == 1
        out = dict(_record("combine"))
        out["title"] = "Mechanistically distinct evolved replacement"
        out["statement"] = "A distinct evolved replacement hypothesis."
        return out

    monkeypatch.setattr(agent, "_generate_dedup_replacement", replacement)
    task = Task(
        id="tsk_evolution_dedup",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="evolution",
        action="EvolveTopHypotheses",
        payload={"top_k": 2, "strategies": ["combine"]},
    )

    result = await agent.execute(task)

    assert result.hypothesis_ids == ["hyp_replacement"]
    assert result.extra["dedup_replacement_attempts"] == 2
    assert result.extra["dedup_replacements"][0]["duplicate_id"] == "hyp_duplicate"
    assert result.extra["dedup_replacements"][-1]["accepted"] is True
    assert result.extra["failed_strategies"] == []


@pytest.mark.asyncio
async def test_evolution_dedup_replacement_requires_capability_validation(
    monkeypatch, conn, tmp_cfg
) -> None:
    session = _session()
    captured: dict = {}

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        captured.update(kwargs)
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "name": "record_hypothesis",
                        "input": _record("combine"),
                    }
                ],
            ),
            iterations=4,
        )

    monkeypatch.setattr(evolution_mod, "run_tool_loop", fake_tool_loop)
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = [
        {"name": "capability_search", "input_schema": {}},
        {"name": "capability_get", "input_schema": {}},
        {"name": "capability_validate_workflow", "input_schema": {}},
    ]
    agent = EvolutionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=MagicMock(), tools=tools))
    attempt = evolution_mod.EvolutionAttempt(
        strategy="combine",
        mode_for_route="combine",
        prompt="Combine the parent mechanisms.",
        parent_ids=["hyp_a", "hyp_b"],
    )

    record = await agent._generate_dedup_replacement(
        session=session,
        attempt=attempt,
        task_id="tsk_dedup_grounding",
        seen_urls=set(),
        rejected_record=_record("combine"),
        duplicate=None,
        duplicate_id="hyp_duplicate",
        attempt_n=1,
    )

    assert record is not None
    assert captured["force_terminal_tool"] == "record_hypothesis"
    assert captured["terminal_required_tool_names"] == (
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
    )
    assert captured["ctx"].mode == "combine_dedup_replacement"
    assert captured["spec"].tool_choice == {"type": "auto"}
