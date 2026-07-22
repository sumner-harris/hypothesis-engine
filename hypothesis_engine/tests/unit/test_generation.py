"""Tests for GenerationAgent recovery behavior."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hypothesis_engine.agents.generation as generation_mod
from hypothesis_engine.agents.base import AgentDeps
from hypothesis_engine.agents.generation import GenerationAgent
from hypothesis_engine.llm.anthropic_client import AnthropicResponse
from hypothesis_engine.llm.tool_loop import ToolLoopResult
from hypothesis_engine.models import Hypothesis, ResearchPlan, Session, Task
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import sessions as sess_repo
from hypothesis_engine.storage.repos import tasks as task_repo


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
        transcript_id="trn_generation",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _study_plan(component_ids: list[str] | None = None) -> list[dict]:
    ids = component_ids or [
        "primary_method",
        "validation",
        "explanation",
        "synthesis_intervention",
        "characterization_validation",
        "theory_modeling",
        "quantitative_decision_criteria",
        "controls_failure_modes",
    ]
    return [
        {
            "component_id": component_id,
            "component_label": component_id.replace("_", " ").title(),
            "role": "primary_driver" if index == 0 else "supporting",
            "objective": f"Make the {component_id} component executable.",
            "methods": ["Run AIMD collision-cascade simulations at 10-80 eV."],
            "variables": ["ion energy", "incidence angle", "substrate"],
            "outputs": ["top-layer displacement probability", "bottom-layer damage rate"],
            "quantitative_targets": [
                "support if top-layer displacement exceeds bottom-layer damage"
            ],
            "controls_or_comparators": ["free-standing monolayer", "inert ion control"],
            "failure_criteria": ["bottom-layer damage dominates across the sweep"],
        }
        for index, component_id in enumerate(ids)
    ]


def _record() -> dict:
    return {
        "title": "Threshold-mediated chalcogen defect nucleation",
        "statement": (
            "Low-energy ions create monolayer TMD defects when local "
            "chalcogen recoil thresholds are transiently lowered."
        ),
        "mechanism": (
            "Ion impact couples to out-of-plane chalcogen modes, concentrating "
            "momentum into bonds adjacent to pre-strained lattice sites."
        ),
        "entities": ["low energy ion", "monolayer TMD", "chalcogen vacancy"],
        "anticipated_outcomes": (
            "Vacancy yield should increase at strained regions and show a sharp "
            "dependence on ion energy."
        ),
        "novelty_argument": (
            "The hypothesis emphasizes dynamic threshold lowering rather than "
            "only static displacement energy."
        ),
        "study_plan": _study_plan(),
        "citations": [],
    }


def test_hypothesis_record_error_requires_profile_study_plan_components() -> None:
    record = _record()
    record["study_plan"] = _study_plan(["mechanism_model"])

    error = generation_mod._hypothesis_record_error(
        record,
        required_study_plan_components=["mechanism_model", "validation_plan"],
    )

    assert error == "study_plan missing required component_id(s): validation_plan"


def test_hypothesis_record_error_accepts_all_required_study_plan_components() -> None:
    record = _record()
    record["study_plan"] = _study_plan(["mechanism_model", "validation_plan"])

    error = generation_mod._hypothesis_record_error(
        record,
        required_study_plan_components=["mechanism_model", "validation_plan"],
    )

    assert error is None


def test_ensure_hypothesis_title_derives_from_statement_before_validation() -> None:
    record = _record()
    record.pop("title")

    normalized = generation_mod._ensure_hypothesis_title(record)
    error = generation_mod._hypothesis_record_error(
        record,
        required_study_plan_components=["primary_method", "validation", "explanation"],
    )

    assert normalized is True
    assert record["title"].startswith("Low-energy ions create monolayer TMD defects")
    assert error is None


async def _seed_session(conn) -> Session:
    now = datetime.now(UTC)
    session = Session(
        id="ses_generation",
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
    await sess_repo.insert(conn, session)
    return session


@pytest.mark.asyncio
async def test_generation_rag_workflow_discovers_waits_then_debates(
    monkeypatch, conn, tmp_cfg
) -> None:
    tmp_cfg.rag.enabled = True
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    session = await _seed_session(conn)
    calls: list[dict] = []
    wait_seen: list[str] = []

    discovery_record = {
        "search_summary": "Found ion implantation and Janus TMD synthesis papers.",
        "promising_directions": ["implantation-assisted Janus conversion"],
        "knowledge_gaps": ["few direct monolayer Janus implantation studies"],
        "recommended_retrieval_queries": ["ion implantation Janus TMD synthesis"],
        "capability_application_evidence": [
            {
                "capability_id": "sim:dft:vasp-01",
                "version": "placeholder-0.1",
                "intended_use": "Calculate defect formation energies.",
                "literature_queries": ["VASP defect formation energy monolayer TMD"],
                "evidence_summary": (
                    "Prior DFT studies use defect formation energies to rank vacancy states."
                ),
                "observables": ["defect formation energy"],
                "limitations": ["Requires converged supercell and chemical potentials"],
            }
        ],
        "capability_gaps": ["No catalog-backed ion implantation capability."],
    }

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        tool_names = [tool.get("name") for tool in kwargs["spec"].tools]
        calls.append(
            {
                "ctx_mode": kwargs["ctx"].mode,
                "tool_names": tool_names,
                "force_terminal_tool": kwargs.get("force_terminal_tool"),
                "terminal_tool_names": kwargs.get("terminal_tool_names"),
                "terminal_required_tool_names": kwargs.get("terminal_required_tool_names"),
                "prompt": kwargs["spec"].user_blocks[0].text,
            }
        )
        if len(calls) == 1:
            assert "record_literature_discovery" in tool_names
            assert "record_hypothesis" not in tool_names
            assert "rag_retrieve_context" not in tool_names
            assert "capability_search" in tool_names
            assert "capability_get" in tool_names
            assert "capability_validate_workflow" not in tool_names
            assert "# Capability-application evidence requirement" in calls[-1]["prompt"]
            assert "capability_application_evidence" in calls[-1]["prompt"]
            return ToolLoopResult(
                response=_response(
                    stop_reason="tool_use",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": "discovery",
                            "name": "record_literature_discovery",
                            "input": discovery_record,
                        }
                    ],
                ),
                iterations=2,
                tool_calls=[
                    {
                        "name": "capability_search",
                        "args": {"query": "TMD defect calculations"},
                        "is_error": False,
                    },
                    {
                        "name": "capability_get",
                        "args": {"capability_ids": ["sim:dft:vasp-01"]},
                        "is_error": False,
                    },
                    {
                        "name": "arxiv_search",
                        "args": {"query": "VASP defect formation energy monolayer TMD"},
                        "is_error": False,
                    },
                ],
                seen_urls={"https://arxiv.org/pdf/2401.00001v1"},
            )
        assert wait_seen == [session.id]
        assert "record_hypothesis" in tool_names
        assert "rag_retrieve_context" in tool_names
        assert "rag_kb_status" in tool_names
        assert "capability_search" in tool_names
        assert "capability_get" in tool_names
        assert "capability_validate_workflow" in tool_names
        assert "arxiv_search" not in tool_names
        assert "chemrxiv_search" not in tool_names
        assert "web_fetch" not in tool_names
        assert "Literature discovery map" in calls[-1]["prompt"]
        assert "# Primary author lens for complete-study synthesis" in calls[-1]["prompt"]
        assert "Characterization and validation" in calls[-1]["prompt"]
        assert "Lens-informed retrieval angles" in calls[-1]["prompt"]
        assert "Required complete-study elements" in calls[-1]["prompt"]
        assert "Required structured study_plan work packages" in calls[-1]["prompt"]
        assert "study_plan" in calls[-1]["prompt"]
        assert "complete study concept" in calls[-1]["prompt"]
        assert "## Capability application evidence" in calls[-1]["prompt"]
        assert "sim:dft:vasp-01" in calls[-1]["prompt"]
        assert "VASP defect formation energy monolayer TMD" in calls[-1]["prompt"]
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "hypothesis",
                        "name": "record_hypothesis",
                        "input": _record(),
                    }
                ],
            ),
            iterations=3,
            tool_calls=[{"name": "rag_retrieve_context", "is_error": False}],
            seen_urls={"https://example.test/rag-source.pdf"},
        )

    async def fake_wait(_cfg, session_id: str) -> dict:
        wait_seen.append(session_id)
        return {
            "enabled": True,
            "waited": True,
            "paper_count": 2,
            "indexed_paper_count": 2,
        }

    monkeypatch.setattr(generation_mod, "run_tool_loop", fake_tool_loop)
    monkeypatch.setattr(generation_mod, "_wait_for_rag_ingest_after_discovery", fake_wait)
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = [
        {"name": "arxiv_search", "description": "", "input_schema": {}},
        {"name": "chemrxiv_search", "description": "", "input_schema": {}},
        {"name": "web_fetch", "description": "", "input_schema": {}},
        {"name": "rag_kb_status", "description": "", "input_schema": {}},
        {"name": "rag_retrieve_context", "description": "", "input_schema": {}},
        {"name": "capability_search", "description": "", "input_schema": {}},
        {"name": "capability_get", "description": "", "input_schema": {}},
        {
            "name": "capability_validate_workflow",
            "description": "",
            "input_schema": {},
        },
    ]

    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=MagicMock(), tools=tools))
    task = Task(
        id="tsk_generation_rag",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={
            "strategy": "literature",
            "n": 1,
            "discovery_profile": generation_mod.initial_discovery_profile_for_index(2),
        },
    )

    result = await agent.execute(task)

    assert result.kind == "hypothesis_created"
    assert result.extra["generation_workflow"] == "rag_discovery_debate"
    assert result.extra["discovery_iterations"] == 2
    assert result.extra["debate_iterations"] == 3
    assert result.extra["rag_ingest_wait"]["indexed_paper_count"] == 2
    assert calls[0]["ctx_mode"] == "literature_discovery"
    assert calls[1]["ctx_mode"] == "debate"
    assert calls[0]["force_terminal_tool"] == "record_literature_discovery"
    assert calls[1]["force_terminal_tool"] == "record_hypothesis"
    assert calls[0]["terminal_required_tool_names"] == (
        "capability_search",
        "capability_get",
    )
    assert calls[1]["terminal_required_tool_names"] == (
        "capability_search",
        "capability_get",
        "capability_validate_workflow",
    )


@pytest.mark.asyncio
async def test_generation_rag_initial_discovery_barrier_merges_parallel_tasks(
    monkeypatch, conn, tmp_cfg
) -> None:
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.generation_wait_timeout_seconds = 5
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    monkeypatch.setattr(generation_mod, "_INITIAL_DISCOVERY_POLL_SECONDS", 0.01)
    session = await _seed_session(conn)
    now = datetime.now(UTC)
    initial_tasks = [
        Task(
            id="tsk_generation_rag_biorxiv",
            session_id=session.id,
            created_at=now,
            agent="generation",
            action="CreateInitialHypotheses",
            payload={
                "strategy": "literature",
                "n": 1,
                "initial_generation": True,
                "initial_index": 0,
                "initial_total": 2,
                "discovery_group": "initial",
                "discovery_profile": generation_mod.initial_discovery_profile_for_index(0),
            },
        ),
        Task(
            id="tsk_generation_rag_chemrxiv",
            session_id=session.id,
            created_at=now + timedelta(seconds=1),
            agent="generation",
            action="CreateInitialHypotheses",
            payload={
                "strategy": "literature",
                "n": 1,
                "initial_generation": True,
                "initial_index": 1,
                "initial_total": 2,
                "discovery_group": "initial",
                "discovery_profile": generation_mod.initial_discovery_profile_for_index(1),
            },
        ),
    ]
    expected_profiles = {task.id: task.payload["discovery_profile"] for task in initial_tasks}
    for queued_task in initial_tasks:
        assert await task_repo.enqueue(conn, queued_task) is True

    discovery_records = {
        "tsk_generation_rag_biorxiv": {
            "search_summary": "bioRxiv microbial nylonase papers emphasize nylon-6,6 selectivity.",
            "promising_directions": ["screen microbial amidases against mixed nylon waste"],
            "knowledge_gaps": ["few assays compare nylon-6 and nylon-6,6 in one mixture"],
            "recommended_retrieval_queries": ["microbial nylonase nylon 6,6 selectivity bioRxiv"],
        },
        "tsk_generation_rag_chemrxiv": {
            "search_summary": "ChemRxiv polymer recycling papers emphasize oligomer repolymerization.",
            "promising_directions": ["capture defined adipate-hexamethylenediamine oligomers"],
            "knowledge_gaps": ["product distributions are rarely linked to repolymerization"],
            "recommended_retrieval_queries": [
                "nylon oligomer repolymerization depolymerization ChemRxiv"
            ],
        },
    }
    discovery_tools = {
        "tsk_generation_rag_biorxiv": "biorxiv_search",
        "tsk_generation_rag_chemrxiv": "chemrxiv_search",
    }
    discovery_urls = {
        "tsk_generation_rag_biorxiv": "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1.full.pdf",
        "tsk_generation_rag_chemrxiv": "https://chemrxiv.org/engage/chemrxiv/article-details/example.pdf",
    }
    discovery_calls: list[str] = []
    debate_prompts: dict[str, str] = {}
    wait_checks: list[str] = []

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        mode = kwargs["ctx"].mode
        task_id = kwargs["ctx"].task_id
        if mode == "literature_discovery":
            discovery_calls.append(task_id)
            prompt = kwargs["spec"].user_blocks[0].text
            profile = expected_profiles[task_id]
            assert "Assigned discovery lens" in prompt
            assert profile["label"] in prompt
            assert profile["id"] in prompt
            return ToolLoopResult(
                response=_response(
                    stop_reason="tool_use",
                    blocks=[
                        {
                            "type": "tool_use",
                            "id": f"discovery-{task_id}",
                            "name": "record_literature_discovery",
                            "input": discovery_records[task_id],
                        }
                    ],
                ),
                iterations=2,
                tool_calls=[{"name": discovery_tools[task_id], "is_error": False}],
                seen_urls={discovery_urls[task_id]},
            )

        assert mode == "debate"
        prompt = kwargs["spec"].user_blocks[0].text
        assert "Combined Literature Discovery Map" in prompt
        assert "# Primary author lens for complete-study synthesis" in prompt
        assert expected_profiles[task_id]["label"] in prompt
        assert "Lens-informed retrieval angles" in prompt
        assert "Required complete-study elements" in prompt
        assert "Required structured study_plan work packages" in prompt
        assert "component_id` exactly as written" in prompt
        assert "payload is invalid if it includes only the primary lens package" in prompt
        assert "complete study hypothesis" in prompt
        assert "Mechanism and causal pathways" in prompt
        assert "Synthesis or intervention routes" in prompt
        assert "bioRxiv microbial nylonase papers" in prompt
        assert "ChemRxiv polymer recycling papers" in prompt
        debate_prompts[task_id] = prompt
        record = dict(_record())
        record["title"] = f"Merged discovery hypothesis {len(debate_prompts)}"
        record["statement"] = f"Merged discovery generated hypothesis {len(debate_prompts)}."
        return ToolLoopResult(
            response=_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": f"hypothesis-{task_id}",
                        "name": "record_hypothesis",
                        "input": record,
                    }
                ],
            ),
            iterations=3,
            tool_calls=[{"name": "rag_retrieve_context", "is_error": False}],
            seen_urls={"https://example.test/rag-source.pdf"},
        )

    async def fake_wait(_cfg, session_id: str) -> dict:
        root = tmp_cfg.session_artifact_dir(session.id) / "generation_discovery"
        first_path = root / "tsk_generation_rag_biorxiv.json"
        second_path = root / "tsk_generation_rag_chemrxiv.json"
        assert first_path.is_file()
        assert second_path.is_file()
        payloads = [json.loads(first_path.read_text()), json.loads(second_path.read_text())]
        assert {payload["task_id"] for payload in payloads} == {
            "tsk_generation_rag_biorxiv",
            "tsk_generation_rag_chemrxiv",
        }
        assert {payload["discovery_profile"]["id"] for payload in payloads} == {
            "mechanism",
            "synthesis_route",
        }
        wait_checks.append(session_id)
        return {
            "enabled": True,
            "waited": True,
            "pending_background_ingest": False,
            "indexed_paper_count": 2,
            "release_reason": "all_background_ingest_completed",
        }

    monkeypatch.setattr(generation_mod, "run_tool_loop", fake_tool_loop)
    monkeypatch.setattr(generation_mod, "_wait_for_rag_ingest_after_discovery", fake_wait)
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = [
        {"name": "biorxiv_search", "description": "", "input_schema": {}},
        {"name": "chemrxiv_search", "description": "", "input_schema": {}},
        {"name": "web_fetch", "description": "", "input_schema": {}},
        {"name": "rag_kb_status", "description": "", "input_schema": {}},
        {"name": "rag_retrieve_context", "description": "", "input_schema": {}},
    ]

    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=MagicMock(), tools=tools))

    results = await asyncio.gather(*(agent.execute(task) for task in initial_tasks))

    assert {result.kind for result in results} == {"hypothesis_created"}
    assert set(discovery_calls) == {
        "tsk_generation_rag_biorxiv",
        "tsk_generation_rag_chemrxiv",
    }
    assert set(debate_prompts) == {
        "tsk_generation_rag_biorxiv",
        "tsk_generation_rag_chemrxiv",
    }
    assert wait_checks == [session.id, session.id]
    for result in results:
        barrier = result.extra["initial_discovery_barrier"]
        assert barrier["release_reason"] == "all_discoveries_completed"
        assert barrier["expected_discoveries"] == 2
        assert barrier["completed_discoveries"] == 2
        assert result.extra["initial_discovery_completed"] == 2
        assert len(result.extra["initial_discovery_discoveries"]) == 2
        assert {
            item["discovery_profile"]["id"]
            for item in result.extra["initial_discovery_discoveries"]
        } == {"mechanism", "synthesis_route"}
        assert {call["name"] for call in result.extra["discovery_tool_calls"]} == {
            "biorxiv_search",
            "chemrxiv_search",
        }


@pytest.mark.asyncio
async def test_generation_rag_wait_releases_when_seed_kb_is_ready(monkeypatch, tmp_cfg) -> None:
    from hypothesis_engine.tools import rag as rag_mod

    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.generation_wait_min_indexed_papers = 3
    tmp_cfg.rag.generation_wait_timeout_seconds = 60
    calls = 0

    def fake_pending(_cfg, _session_id):
        return True

    async def fake_wait_step(_cfg, _session_id, *, timeout_seconds):
        nonlocal calls
        calls += 1
        assert timeout_seconds == 1.0
        return {
            "enabled": True,
            "pending_background_ingest": calls == 1,
            "active_background_tasks": 1 if calls == 1 else 0,
            "paper_count": 20,
            "indexed_paper_count": 0,
            "seed_chunk_count": 1241,
            "seed_kb_ready": True,
            "reserved_paper_count": 17 if calls == 1 else 0,
        }

    monkeypatch.setattr(rag_mod, "background_ingest_pending", fake_pending)
    monkeypatch.setattr(rag_mod, "wait_for_background_ingest_step", fake_wait_step)

    status = await generation_mod._wait_for_rag_ingest_after_discovery(tmp_cfg, "ses")

    assert calls == 1
    assert status["waited"] is True
    assert status["released_early"] is True
    assert status["release_reason"] == "seed_kb_ready"
    assert status["generation_wait_min_indexed_papers"] == 3


@pytest.mark.asyncio
async def test_generation_persists_missing_title_without_recovery(conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    record = _record()
    record.pop("title")
    loop_result = ToolLoopResult(
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
    llm = MagicMock()
    llm.call = AsyncMock()
    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    agent._persist_detail = AsyncMock(return_value=generation_mod.PersistResult("hyp_first", True))
    task = Task(
        id="tsk_generation_missing_title",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={"strategy": "literature", "n": 1},
    )

    result = await agent._persist_recorded_hypothesis(
        session=session,
        task=task,
        prompt="original prompt with full discovery map",
        loop_result=loop_result,
        seen_urls=set(),
        strategy="literature",
        expected_study_plan_components=["primary_method", "validation", "explanation"],
        extra={},
    )

    persisted_record = agent._persist_detail.await_args.args[1]
    assert result.hypothesis_ids == ["hyp_first"]
    assert result.extra["recovered_record_hypothesis"] is False
    assert result.extra["normalized_missing_title"] is True
    assert persisted_record["title"].startswith("Low-energy ions create monolayer TMD defects")
    llm.call.assert_not_called()


@pytest.mark.asyncio
async def test_generation_supplements_stored_citations_from_rag_sources(conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    record = _record()
    record["citations"] = [{"title": "Explicit source", "url": "https://arxiv.org/pdf/1111.1111v1"}]
    loop_result = ToolLoopResult(
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
        seen_urls={
            "https://arxiv.org/pdf/1111.1111v1",
            "https://arxiv.org/pdf/2222.2222v1",
            "https://arxiv.org/pdf/3333.3333v1",
        },
        citation_candidates=[
            {"title": "Explicit duplicate", "url": "https://arxiv.org/pdf/1111.1111v1"},
            {"title": "RAG source A", "url": "https://arxiv.org/pdf/2222.2222v1"},
            {"title": "RAG source B", "url": "https://arxiv.org/pdf/3333.3333v1"},
            {"title": "Unseen source", "url": "https://arxiv.org/pdf/4444.4444v1"},
        ],
    )
    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=MagicMock(), tools=MagicMock()))
    agent._persist_detail = AsyncMock(return_value=generation_mod.PersistResult("hyp_cited", True))
    task = Task(
        id="tsk_generation_citation_candidates",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={"strategy": "literature", "n": 1},
    )

    result = await agent._persist_recorded_hypothesis(
        session=session,
        task=task,
        prompt="original prompt",
        loop_result=loop_result,
        seen_urls=loop_result.seen_urls,
        strategy="literature",
        expected_study_plan_components=["primary_method", "validation", "explanation"],
        extra={},
    )

    persisted_record = agent._persist_detail.await_args.args[1]
    assert result.hypothesis_ids == ["hyp_cited"]
    assert [(c["title"], c["url"]) for c in persisted_record["citations"]] == [
        ("Explicit source", "https://arxiv.org/pdf/1111.1111v1"),
        ("RAG source A", "https://arxiv.org/pdf/2222.2222v1"),
        ("RAG source B", "https://arxiv.org/pdf/3333.3333v1"),
    ]


@pytest.mark.asyncio
async def test_generation_repairs_study_plan_from_final_table_without_recovery(
    conn, tmp_cfg
) -> None:
    session = await _seed_session(conn)
    required_components = [
        "synthesis_intervention",
        "characterization_validation",
        "theory_modeling",
        "quantitative_decision_criteria",
        "controls_failure_modes",
    ]
    record = _record()
    record["study_plan"] = _study_plan(["synthesis_intervention"])
    final_text = """
HYPOTHESIS

| Component ID | Label | Role | Objective | Methods | Variables | Outputs | Quantitative Targets | Controls/Comparators | Failure Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `synthesis_intervention` | Synthesis Route | primary_driver | Create Janus MoSSe | Se+ implantation | Energy | Samples | Top substitution >70% | Pristine MoS2 | Amorphization |
| `characterization_validation` | Validation | validation | Confirm Janus structure | Raman and HAADF-STEM | Laser wavelength | Raman shifts | Symmetry mode present | Random alloy | No Janus mode |
| `theory_modeling` | Theory/Modeling | support | Predict window | AIMD impact sweep | Ion energy | Psub and Psput | eta > 2 | Self implantation | eta < 1 |
| `quantitative_decision_criteria` | Decision Criteria | validation | Decide support | Cross-correlate outputs | Raman and STEM metrics | Pass/fail call | all thresholds pass | Baseline MoS2 | random alloy |
| `controls_failure_modes` | Failure Modes | falsification | Bound mechanism | High-energy control | Dose | Damage map | high-energy damage | 1 keV control | no energy dependence |
"""
    loop_result = ToolLoopResult(
        response=_response(
            stop_reason="tool_use",
            blocks=[
                {"type": "text", "text": final_text},
                {
                    "type": "tool_use",
                    "id": "hypothesis",
                    "name": "record_hypothesis",
                    "input": record,
                },
            ],
        ),
        iterations=1,
        tool_calls=[],
        seen_urls=set(),
    )
    llm = MagicMock()
    llm.call = AsyncMock()
    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=MagicMock()))
    agent._persist_detail = AsyncMock(
        return_value=generation_mod.PersistResult("hyp_repaired", True)
    )
    task = Task(
        id="tsk_generation_repair_study_plan",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={"strategy": "literature", "n": 1},
    )

    result = await agent._persist_recorded_hypothesis(
        session=session,
        task=task,
        prompt="original prompt with full discovery map",
        loop_result=loop_result,
        seen_urls=set(),
        strategy="literature",
        expected_study_plan_components=required_components,
        extra={},
    )

    persisted_record = agent._persist_detail.await_args.args[1]
    persisted_components = [item["component_id"] for item in persisted_record["study_plan"]]
    assert result.hypothesis_ids == ["hyp_repaired"]
    assert result.extra["recovered_record_hypothesis"] is False
    assert result.extra["repaired_study_plan_from_text"] is True
    assert persisted_components == required_components
    assert persisted_record["study_plan"][1]["methods"] == ["Raman and HAADF-STEM"]
    llm.call.assert_not_called()


@pytest.mark.asyncio
async def test_generation_recovers_missing_record_hypothesis(monkeypatch, conn, tmp_cfg) -> None:
    tmp_cfg.vectors.dedup_cosine_threshold = 2.0
    session = await _seed_session(conn)
    captured_specs: list = []

    async def fake_tool_loop(*_args, **kwargs) -> ToolLoopResult:
        assert kwargs["terminal_min_seen_urls"] == 0
        assert kwargs["terminal_requirement_hint"] is None
        captured_specs.append(kwargs["spec"])
        return ToolLoopResult(
            response=_response(
                stop_reason="end_turn",
                blocks=[
                    {
                        "type": "text",
                        "text": "Here is a plausible hypothesis, but no tool call.",
                    }
                ],
            ),
            iterations=1,
            tool_calls=[],
            seen_urls=set(),
        )

    monkeypatch.setattr(generation_mod, "run_tool_loop", fake_tool_loop)
    llm = MagicMock()
    llm.call = AsyncMock(
        return_value=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "recover_hypothesis",
                    "name": "record_hypothesis",
                    "input": _record(),
                }
            ],
        )
    )
    tools = MagicMock()
    tools.anthropic_tools_for.return_value = [
        {"name": "pubmed_search", "description": "", "input_schema": {}},
        {"name": "arxiv_search", "description": "", "input_schema": {}},
        {"name": "web_fetch", "description": "", "input_schema": {}},
    ]

    agent = GenerationAgent(AgentDeps(cfg=tmp_cfg, db=conn, llm=llm, tools=tools))
    task = Task(
        id="tsk_generation",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={"strategy": "literature", "n": 1},
    )

    result = await agent.execute(task)

    assert result.kind == "hypothesis_created"
    assert len(result.hypothesis_ids) == 1
    assert result.extra["recovered_record_hypothesis"] is True
    assert result.extra["recovery_attempts"] == 1
    assert (
        result.extra["recovery_max_output_tokens"]
        == tmp_cfg.generation.hypothesis_recovery_max_output_tokens
    )
    assert result.extra["recovery_reason"] == "missing record_hypothesis tool call"
    prompt = captured_specs[0].user_blocks[0].text
    assert "web_search" not in prompt
    assert "pubmed_search" in prompt
    assert "arxiv_search" in prompt
    spec = llm.call.await_args.args[0]
    assert spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    assert spec.max_output_tokens == tmp_cfg.generation.hypothesis_recovery_max_output_tokens
    ctx = llm.call.await_args.args[1]
    assert ctx.task_id == task.id


@pytest.mark.asyncio
async def test_generation_persist_respects_max_ideas_cap(conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    tmp_cfg.run.max_ideas = 1
    await hyp_repo.insert(
        conn,
        Hypothesis(
            id="hyp_existing",
            session_id=session.id,
            created_at=datetime.now(UTC),
            created_by="generation",
            strategy="literature",
            title="existing",
            summary="existing",
            full_text="existing",
            artifact_path="artifacts/existing.md",
            state="draft",
        ),
    )

    agent = GenerationAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )

    _hid, was_new = await agent._persist(session.id, _record(), strategy="literature")

    assert was_new is False
    assert await hyp_repo.count_for_session(conn, session.id) == 1


@pytest.mark.asyncio
async def test_generation_replaces_dedup_duplicate(monkeypatch, conn, tmp_cfg) -> None:
    session = await _seed_session(conn)
    tmp_cfg.generation.dedup_replacement_attempts = 2
    await hyp_repo.insert(
        conn,
        Hypothesis(
            id="hyp_duplicate",
            session_id=session.id,
            created_at=datetime.now(UTC),
            created_by="generation",
            strategy="literature",
            title="Duplicate hypothesis",
            summary="A duplicate mechanism to avoid.",
            full_text="duplicate",
            artifact_path="artifacts/duplicate.md",
            state="draft",
        ),
    )
    loop_result = ToolLoopResult(
        response=_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "hypothesis",
                    "name": "record_hypothesis",
                    "input": _record(),
                }
            ],
        ),
        iterations=1,
        tool_calls=[],
        seen_urls=set(),
    )
    agent = GenerationAgent(
        AgentDeps(
            cfg=tmp_cfg,
            db=conn,
            llm=MagicMock(),
            tools=MagicMock(),
        )
    )
    agent._persist_detail = AsyncMock(
        side_effect=[
            generation_mod.PersistResult(
                "hyp_duplicate", False, reason="dedup_duplicate", duplicate_id="hyp_duplicate"
            ),
            generation_mod.PersistResult("hyp_replacement", True),
        ]
    )

    async def replacement(**kwargs):
        assert kwargs["duplicate"].title == "Duplicate hypothesis"
        assert kwargs["attempt_n"] == 1
        out = dict(_record())
        out["title"] = "Mechanistically distinct replacement"
        out["statement"] = "A distinct replacement hypothesis."
        return out

    monkeypatch.setattr(agent, "_generate_dedup_replacement", replacement)
    task = Task(
        id="tsk_generation_dedup",
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="generation",
        action="CreateInitialHypotheses",
        payload={"strategy": "literature", "n": 1},
    )

    result = await agent._persist_recorded_hypothesis(
        session=session,
        task=task,
        prompt="original prompt",
        loop_result=loop_result,
        seen_urls=set(),
        strategy="literature",
        expected_study_plan_components=["primary_method", "validation", "explanation"],
        extra={},
    )

    assert result.hypothesis_ids == ["hyp_replacement"]
    assert result.extra["dedup_replacement_attempts"] == 2
    assert result.extra["dedup_replacement_attempts_used"] == 1
    assert result.extra["dedup_replaced"] is True
    assert result.extra["dedup_rejections"][0]["duplicate_id"] == "hyp_duplicate"


def test_initial_discovery_profile_for_index_uses_configured_yaml(tmp_path) -> None:
    profile_path = tmp_path / "biology_profiles.yaml"
    profile_path.write_text(
        """required_work_packages:
  - id: assay_plan
    label: Assay plan
    objective: Specify concrete assays and controls.
    expected_outputs:
      - readouts
      - controls
profiles:
  - id: pathway
    label: Pathway biology
    objective: Focus on biological pathways and causal molecular steps.
    search_guidance:
      - Prefer pathway and perturbation terms.
    avoid_overfocus: Do not focus only on clinical reports.
    suggested_query_angles:
      - signaling pathway
  - id: assay
    label: Assay validation
    objective: Focus on assays, controls, and measurable readouts.
""",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(generation=SimpleNamespace(discovery_profiles=str(profile_path)))

    first = generation_mod.initial_discovery_profile_for_index(0, cfg)
    second = generation_mod.initial_discovery_profile_for_index(1, cfg)
    third = generation_mod.initial_discovery_profile_for_index(2, cfg)

    assert first["id"] == "pathway"
    assert first["label"] == "Pathway biology"
    assert first["search_guidance"] == ["Prefer pathway and perturbation terms."]
    assert first["suggested_query_angles"] == ["signaling pathway"]
    assert first["required_work_packages"][0]["id"] == "assay_plan"
    assert first["required_work_packages"][0]["expected_outputs"] == ["readouts", "controls"]
    assert second["id"] == "assay"
    assert third["id"] == "pathway"


def test_bundled_discovery_profile_sets_load() -> None:
    expected = {
        "materials_chemistry.yaml": [
            "mechanism",
            "synthesis_route",
            "characterization",
            "theory_modeling",
            "adjacent_analogs",
            "constraints_negative",
        ],
        "cs_ai.yaml": [
            "algorithms_architecture",
            "data_training_signal",
            "evaluation_benchmarks",
            "systems_efficiency",
            "robustness_safety",
            "adjacent_transfer",
        ],
        "microbiology.yaml": [
            "microbial_mechanism",
            "strains_ecology",
            "assays_phenotyping",
            "engineering_intervention",
            "genomics_evolution",
            "biosafety_limits",
        ],
    }
    profile_dir = generation_mod.PROJECT_ROOT / "config" / "discovery_profiles"

    for filename, ids in expected.items():
        cfg = SimpleNamespace(
            generation=SimpleNamespace(discovery_profiles=str(profile_dir / filename))
        )
        profiles = generation_mod.initial_discovery_profiles(cfg)

        assert [profile["id"] for profile in profiles] == ids
        assert all(profile["label"] for profile in profiles)
        assert all(profile["objective"] for profile in profiles)
        assert all(profile["primary_driver_guidance"] for profile in profiles)
        assert all(profile["required_study_elements"] for profile in profiles)
        assert all(profile["required_work_packages"] for profile in profiles)


def test_literature_discovery_prompt_includes_assigned_diversity_lens() -> None:
    profile = generation_mod.initial_discovery_profile_for_index(2)
    prompt = generation_mod._literature_discovery_prompt(
        ResearchPlan(objective="Find microbial nylonase mechanisms."),
        [{"name": "biorxiv_search", "description": "", "input_schema": {}}],
        discovery_profile=profile,
    )

    assert "# Assigned discovery lens" in prompt
    assert "Characterization and validation" in prompt
    assert "characterization" in prompt
    assert "Other initial generation tasks are covering different lenses" in prompt
    assert "assigned_perspective" in prompt
    assert "queries_run" in prompt


def test_literature_discovery_prompt_conditions_searches_on_capabilities() -> None:
    prompt = generation_mod._literature_discovery_prompt(
        ResearchPlan(objective="Characterize defects in graphene."),
        [
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {
                "name": "capability_search",
                "description": "",
                "input_schema": {},
            },
            {"name": "capability_get", "description": "", "input_schema": {}},
        ],
    )

    assert "# Capability-application evidence requirement" in prompt
    assert "graphene" in prompt
    assert "micro-Raman spectroscopy" in prompt
    assert "capability_application_evidence" in prompt


def test_discovery_evidence_keeps_only_retrieved_capabilities_and_executed_queries() -> None:
    record = {
        "capability_application_evidence": [
            {
                "capability_id": "sim:dft:vasp-01",
                "literature_queries": [
                    "VASP defect formation energy monolayer TMD",
                    "query that was never executed",
                ],
            },
            {
                "capability_id": "sim:md:lammps-01",
                "literature_queries": ["LAMMPS ion collision cascade TMD"],
            },
        ],
        "capability_gaps": [],
    }
    tool_calls = [
        {
            "name": "capability_get",
            "args": {"capability_ids": ["sim:dft:vasp-01"]},
            "is_error": False,
        },
        {
            "name": "arxiv_search",
            "args": {"query": "VASP defect formation energy monolayer TMD"},
            "is_error": False,
        },
    ]

    generation_mod._sanitize_capability_application_evidence(record, tool_calls)

    assert record["capability_application_evidence"] == [
        {
            "capability_id": "sim:dft:vasp-01",
            "literature_queries": ["VASP defect formation energy monolayer TMD"],
        }
    ]
    gaps = "\n".join(record["capability_gaps"])
    assert "query that was never executed" in gaps
    assert "sim:md:lammps-01" in gaps


def test_recovery_preserves_capability_refs_from_blocked_candidate() -> None:
    source = {
        "study_plan": [
            {
                "component_id": "theory_modeling",
                "capability_refs": [
                    {
                        "capability_id": "sim:dft:vasp-01",
                        "version": "placeholder-0.1",
                        "purpose": "Calculate defect energies.",
                    }
                ],
            }
        ]
    }
    recovered = {
        "study_plan": [
            {
                "component_id": "theory_modeling",
                "methods": ["Run VASP calculations."],
            }
        ]
    }

    generation_mod._preserve_capability_refs(recovered, source)

    assert (
        recovered["study_plan"][0]["capability_refs"] == source["study_plan"][0]["capability_refs"]
    )


def test_literature_discovery_prompt_prefers_biorxiv_for_bio_topics() -> None:
    prompt = generation_mod._literature_discovery_prompt(
        ResearchPlan(objective="Find microbial nylonase mechanisms."),
        [
            {"name": "biorxiv_search", "description": "", "input_schema": {}},
            {"name": "pubmed_search", "description": "", "input_schema": {}},
            {"name": "europe_pmc_search", "description": "", "input_schema": {}},
        ],
    )

    assert "prefer biorxiv_search first" in prompt
    assert "pubmed_search and europe_pmc_search" in prompt


def test_generation_search_provider_guidance_prefers_biorxiv_independently() -> None:
    guidance = generation_mod._search_provider_guidance(
        {"biorxiv_search", "pubmed_search", "europe_pmc_search"}
    )

    assert "`biorxiv_search` first" in guidance
    assert "`pubmed_search` and `europe_pmc_search`" in guidance
    assert "chemrxiv_search" not in guidance
