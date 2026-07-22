"""Tests for ReflectionAgent recovery behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hypothesis_engine.agents.reflection as reflection_mod
from hypothesis_engine.agents.base import AgentDeps
from hypothesis_engine.agents.reflection import ReflectionAgent
from hypothesis_engine.capabilities.models import (
    CapabilityGroundingIssue,
    CapabilityGroundingReport,
)
from hypothesis_engine.config import PROJECT_ROOT
from hypothesis_engine.llm.anthropic_client import AnthropicResponse
from hypothesis_engine.llm.tool_loop import ToolLoopResult
from hypothesis_engine.models import Hypothesis, ResearchPlan, Session, Task
from hypothesis_engine.storage.artifacts import read_json, write_json
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import reviews as rev_repo
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
        transcript_id="trn_reflection",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _session() -> Session:
    now = datetime.now(UTC)
    return Session(
        id="ses_reflection",
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


def _hypothesis(session_id: str) -> Hypothesis:
    return Hypothesis(
        id="hyp_reflection",
        session_id=session_id,
        created_at=datetime.now(UTC),
        created_by="generation",
        strategy="literature",
        title="Ion defect hypothesis",
        summary="summary",
        full_text="Low-energy ions may induce chalcogen vacancies in monolayer TMDs.",
        artifact_path="artifacts/hyp_reflection.md",
        state="draft",
    )


def test_review_record_error_normalizes_common_score_scales() -> None:
    record = {
        "verdict": "missing_piece",
        "kind": "full",
        "novelty": 4,
        "correctness": 3,
        "testability": 4,
        "feasibility": 3,
        "assumptions": [],
        "evidence": [],
        "notes": "Uses a 1-5 scale.",
    }

    assert reflection_mod._review_record_error(record, expected_kind="full") is None
    assert record["novelty"] == 0.8
    assert record["correctness"] == 0.6
    assert record["testability"] == 0.8
    assert record["feasibility"] == 0.6


def test_review_record_error_rejects_unsupported_score_scale() -> None:
    record = {
        "verdict": "neutral",
        "kind": "full",
        "novelty": 101,
        "correctness": 3,
        "testability": 4,
        "feasibility": 3,
        "assumptions": [],
        "evidence": [],
        "notes": "Invalid score scale.",
    }

    assert (
        reflection_mod._review_record_error(record, expected_kind="full")
        == "review scores outside supported range"
    )


@pytest.mark.asyncio
async def test_reflection_recovers_when_record_review_missing(monkeypatch, conn, tmp_cfg) -> None:
    session = _session()
    hypothesis = _hypothesis(session.id)
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, hypothesis)

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        assert kwargs["terminal_min_seen_urls"] == 0
        assert kwargs["terminal_requirement_hint"] is None
        return ToolLoopResult(
            response=_response(
                stop_reason="end_turn",
                blocks=[{"type": "text", "text": "This looks plausible but needs validation."}],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls={"https://example.test/paper"},
        )

    monkeypatch.setattr(reflection_mod, "run_tool_loop", fake_tool_loop)

    recovered_record = {
        "verdict": "missing_piece",
        "kind": "full",
        "novelty": 0.7,
        "correctness": 0.5,
        "testability": 0.8,
        "feasibility": 0.6,
        "assumptions": [],
        "evidence": [
            {
                "claim": "The mechanism needs direct validation.",
                "url": "https://example.test/paper",
                "excerpt": "validation evidence",
            }
        ],
        "notes": "Recovered compact review.",
    }
    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "recover_review",
                    "name": "record_review",
                    "input": recovered_record,
                }
            ],
        )
    )
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = ReflectionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_reflection",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="reflection",
        action="ReviewHypothesis",
        target_id=hypothesis.id,
    )

    result = await agent.execute(task)

    assert result.kind == "review_completed"
    assert result.extra == {
        "verdict": "missing_piece",
        "recovered_record_review": True,
        "recovery_attempts": 1,
        "recovery_max_output_tokens": tmp_cfg.reflection.review_recovery_max_output_tokens,
        "recovery_reason": "missing record_review tool call",
    }
    spec = llm.call.await_args.args[0]
    assert spec.tool_choice == {"type": "tool", "name": "record_review"}
    assert [tool["name"] for tool in spec.tools] == ["record_review"]
    assert spec.max_output_tokens == tmp_cfg.reflection.review_recovery_max_output_tokens

    reviews = await rev_repo.list_for_hypothesis(conn, hypothesis.id)
    assert len(reviews) == 1
    assert reviews[0].verdict == "missing_piece"
    assert reviews[0].scores.novelty == 0.7
    assert reviews[0].scores.feasibility == 0.6
    refreshed = await hyp_repo.fetch(conn, hypothesis.id)
    assert refreshed is not None
    assert refreshed.state == "reviewed"


@pytest.mark.asyncio
async def test_reflection_retries_incomplete_review_with_larger_token_limit(
    monkeypatch, conn, tmp_cfg
) -> None:
    session = _session()
    hypothesis = _hypothesis(session.id)
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, hypothesis)

    async def fake_tool_loop(*_args, **_kwargs) -> ToolLoopResult:
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "bad_review",
                        "name": "record_review",
                        "input": {
                            "verdict": "neutral",
                            "kind": "full",
                            "evidence": [],
                            "notes": "No scores, so this is not complete.",
                        },
                    }
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(reflection_mod, "run_tool_loop", fake_tool_loop)

    valid_record = {
        "verdict": "neutral",
        "kind": "full",
        "novelty": 0.4,
        "correctness": 0.5,
        "testability": 0.6,
        "feasibility": 0.7,
        "assumptions": [],
        "evidence": [],
        "notes": "Recovered complete review.",
    }
    llm = MagicMock()
    llm.call = AsyncMock(
        side_effect=[
            _response(
                stop_reason="max_tokens",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "recover_truncated",
                        "name": "record_review",
                        "input": {"_raw_arguments": '{"verdict":"neutral","kind":"full"'},
                    }
                ],
            ),
            _response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "recover_valid",
                        "name": "record_review",
                        "input": valid_record,
                    }
                ],
            ),
        ]
    )
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = ReflectionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_reflection",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="reflection",
        action="ReviewHypothesis",
        target_id=hypothesis.id,
    )

    result = await agent.execute(task)

    assert result.kind == "review_completed"
    assert result.extra["recovered_record_review"] is True
    assert result.extra["recovery_attempts"] == 2
    assert [call.args[0].max_output_tokens for call in llm.call.await_args_list] == [
        tmp_cfg.reflection.review_recovery_max_output_tokens,
        tmp_cfg.reflection.review_recovery_max_output_tokens
        * tmp_cfg.reflection.review_recovery_token_multiplier,
    ]

    reviews = await rev_repo.list_for_hypothesis(conn, hypothesis.id)
    assert len(reviews) == 1
    assert reviews[0].verdict == "neutral"
    assert reviews[0].scores.novelty == 0.4
    assert reviews[0].scores.correctness == 0.5
    assert reviews[0].scores.testability == 0.6
    assert reviews[0].scores.feasibility == 0.7
    assert "Recovered complete review" in reviews[0].body
    refreshed = await hyp_repo.fetch(conn, hypothesis.id)
    assert refreshed is not None
    assert refreshed.state == "reviewed"


@pytest.mark.asyncio
async def test_reflection_does_not_persist_invalid_review_when_recovery_fails(
    monkeypatch, conn, tmp_cfg
) -> None:
    session = _session()
    hypothesis = _hypothesis(session.id)
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, hypothesis)

    async def fake_tool_loop(*_args, **_kwargs) -> ToolLoopResult:
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "bad_review",
                        "name": "record_review",
                        "input": {"_raw_arguments": "<|tool_call>call:record_review{..."},
                    }
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(reflection_mod, "run_tool_loop", fake_tool_loop)

    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="max_tokens",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "still_bad",
                    "name": "record_review",
                    "input": {"_raw_arguments": '{"verdict":"neutral"'},
                }
            ],
        )
    )
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = []

    agent = ReflectionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_reflection",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="reflection",
        action="ReviewHypothesis",
        target_id=hypothesis.id,
    )

    with pytest.raises(RuntimeError, match="complete record_review"):
        await agent.execute(task)

    assert llm.call.await_count == tmp_cfg.reflection.review_recovery_max_attempts
    reviews = await rev_repo.list_for_hypothesis(conn, hypothesis.id)
    assert reviews == []
    refreshed = await hyp_repo.fetch(conn, hypothesis.id)
    assert refreshed is not None
    assert refreshed.state == "draft"


def test_capability_audit_uses_authoritative_validation_report() -> None:
    record = {
        "capability_audit": {
            "catalog_revision": "invented",
            "status": "ungrounded",
            "validated_capability_ids": [],
            "issues": [
                {
                    "severity": "error",
                    "issue": "missing purpose",
                    "remediation": "rewrite the plan",
                }
            ],
        }
    }
    report = CapabilityGroundingReport(
        catalog_revision="local-placeholder-inventory-v2",
        status="partial",
        referenced_capability_ids=["sim:md:lammps-01"],
        issues=[
            CapabilityGroundingIssue(
                severity="warning",
                code="component_without_capability",
                message="work package has no catalog-backed capability reference",
                component_id="synthesis_intervention",
            )
        ],
    )

    reflection_mod._discard_model_capability_audit(record, report)
    reflection_mod._apply_authoritative_capability_audit(record, report)

    audit = record["capability_audit"]
    assert audit["catalog_revision"] == "local-placeholder-inventory-v2"
    assert audit["status"] == "partial"
    assert audit["validated_capability_ids"] == ["sim:md:lammps-01"]
    assert audit["issues"] == [
        {
            "severity": "warning",
            "code": "component_without_capability",
            "issue": "work package has no catalog-backed capability reference",
            "remediation": (
                "Add an exact catalog-backed capability reference, or retain this as an "
                "explicit local capability gap."
            ),
            "component_id": "synthesis_intervention",
        }
    ]


@pytest.mark.asyncio
async def test_reflection_audits_persisted_plan_without_model_resubmission(
    monkeypatch, conn, tmp_cfg
) -> None:
    session = _session()
    tmp_cfg.capabilities.enabled = True
    tmp_cfg.capabilities.catalog_path = str(PROJECT_ROOT / "config" / "capabilities")
    hypothesis = _hypothesis(session.id)
    artifact_path = await write_json(
        tmp_cfg,
        session.id,
        "hypotheses",
        hypothesis.id,
        {
            "record": {
                "study_plan": [
                    {
                        "component_id": "theory_modeling",
                        "capability_refs": [
                            {
                                "capability_id": "sim:md:lammps-01",
                                "version": "placeholder-0.1",
                                "purpose": "Determine TDEs and simulate collision cascades.",
                                "parameters": [{"name": "temperature", "value": 300, "unit": "K"}],
                            }
                        ],
                    },
                    {
                        "component_id": "synthesis_intervention",
                        "capability_refs": [],
                    },
                ]
            }
        },
    )
    hypothesis = hypothesis.model_copy(update={"artifact_path": artifact_path})
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, hypothesis)

    bogus_model_audit = {
        "catalog_revision": "invented",
        "status": "ungrounded",
        "validated_capability_ids": [],
        "issues": [
            {
                "severity": "error",
                "issue": "Invalid CapabilityReference: missing purpose.",
                "remediation": "Rewrite the study plan.",
            }
        ],
    }
    review_record = {
        "verdict": "missing_piece",
        "kind": "full",
        "novelty": 0.7,
        "correctness": 0.6,
        "testability": 0.8,
        "feasibility": 0.5,
        "assumptions": [],
        "evidence": [],
        "notes": "Ion implantation and FLA remain a local capability gap.",
        "capability_audit": bogus_model_audit,
    }

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        spec = kwargs["spec"]
        tool_names = {tool["name"] for tool in spec.tools}
        assert "capability_validate_workflow" not in tool_names
        assert "capability_audit" not in spec.tools[-1]["input_schema"]["properties"]
        assert kwargs["terminal_required_tool_names"] == (
            "capability_search",
            "capability_get",
        )
        prompt = "\n".join(block.text for block in spec.user_blocks)
        assert "# Authoritative capability audit" in prompt
        assert "local-placeholder-inventory-v2" in prompt
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "review",
                        "name": "record_review",
                        "input": review_record,
                    }
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(reflection_mod, "run_tool_loop", fake_tool_loop)
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = [
        {"name": "arxiv_search", "description": "", "input_schema": {}},
        {"name": "capability_search", "description": "", "input_schema": {}},
        {"name": "capability_get", "description": "", "input_schema": {}},
        {
            "name": "capability_validate_workflow",
            "description": "must be filtered",
            "input_schema": {},
        },
    ]
    agent = ReflectionAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=MagicMock(), tools=tools))
    task = Task(
        id="tsk_reflection_authoritative_audit",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="reflection",
        action="ReviewHypothesis",
        target_id=hypothesis.id,
    )

    result = await agent.execute(task)

    assert result.extra["capability_audit_status"] == "partial"
    reviews = await rev_repo.list_for_hypothesis(conn, hypothesis.id)
    payload = await read_json(tmp_cfg, reviews[0].artifact_path)
    audit = payload["record"]["capability_audit"]
    assert audit["status"] == "partial"
    assert audit["validated_capability_ids"] == ["sim:md:lammps-01"]
    assert {issue["code"] for issue in audit["issues"]} == {
        "capability_availability_uncertain",
        "component_without_capability",
    }
    assert "missing purpose" not in reviews[0].body
