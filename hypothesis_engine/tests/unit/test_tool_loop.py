# Modified from the original work.
"""Tests for the tool-use loop, especially terminal-tool short-circuit.

The terminal-tool short-circuit matters for any provider whose model does NOT
reliably emit `stop_reason="end_turn"` after calling a recording tool — that
includes most OpenAI-compat models (Gemini, OpenAI o-series via tool_calls,
Llama through OpenRouter, etc.). Without the short-circuit they loop until
max_iters and ToolLoopExhausted, even though a perfectly valid record was
emitted on the first call.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hypothesis_engine.llm.anthropic_client import (
    AgentCallSpec,
    AnthropicResponse,
    CachedBlock,
    CallContext,
)
from hypothesis_engine.llm.routing import ModelRoute
from hypothesis_engine.llm.tool_loop import (
    SEARCH_RESULTS_CONTEXT_LIMIT,
    _has_discovery_tool,
    _recordable_citation_candidates,
    _recordable_seen_urls,
    _result_for_model_context,
    _source_key,
    _tool_result_block,
    run_tool_loop,
)


def _fake_response(*, stop_reason: str, blocks: list[dict]) -> AnthropicResponse:
    """Build an AnthropicResponse whose .raw quacks like an anthropic Message."""
    content = []
    for b in blocks:
        # Each block must expose .type, .name, .input, .text
        content.append(
            SimpleNamespace(
                type=b.get("type", "text"),
                name=b.get("name", ""),
                input=b.get("input", {}),
                text=b.get("text", ""),
                id=b.get("id", ""),
            )
        )
    raw = SimpleNamespace(stop_reason=stop_reason, content=content)
    return AnthropicResponse(
        raw=raw,
        transcript_id="trn_x",
        cost_usd=0.0,
        input_tokens=0,
        output_tokens=0,
        cache_read=0,
        cache_write=0,
    )


def _spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[{"name": "search", "description": "", "input_schema": {}}],
        max_output_tokens=512,
    )


def test_literature_searches_count_as_discovery_tools() -> None:
    assert _has_discovery_tool([{"name": "biorxiv_search"}])
    assert _has_discovery_tool([{"name": "chemrxiv_search"}])


def test_rag_retrieve_context_yields_ordered_citation_candidates() -> None:
    result = {
        "is_error": False,
        "content": {
            "source_documents": [
                {"url": "https://arxiv.org/pdf/1111.1111v1", "title": "Primary source"},
                {"url": "https://arxiv.org/pdf/2222.2222v1", "title": "Second source"},
            ],
            "rerank_chunks": [
                {"url": "https://arxiv.org/pdf/1111.1111v1", "title": "Duplicate primary"},
                {"url": "https://arxiv.org/pdf/3333.3333v1", "title": "Third source"},
            ],
        },
    }

    assert _recordable_citation_candidates("rag_retrieve_context", result) == [
        {"url": "https://arxiv.org/pdf/1111.1111v1", "title": "Primary source"},
        {"url": "https://arxiv.org/pdf/2222.2222v1", "title": "Second source"},
        {"url": "https://arxiv.org/pdf/3333.3333v1", "title": "Third source"},
    ]


def _ctx() -> CallContext:
    return CallContext(session_id="s", task_id="t", agent="generation", action="a")


@pytest.mark.asyncio
async def test_loop_ends_on_record_hypothesis_even_when_stop_reason_is_tool_use() -> None:
    """The bug we hit on Gemini: model emits record_hypothesis but keeps
    stop_reason=tool_use, so without short-circuit the loop runs to
    max_iters."""
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1
    # The terminal tool_use is logged but never dispatched.
    assert client.call.await_count == 1
    assert result.tool_calls[0]["name"] == "record_hypothesis"


@pytest.mark.asyncio
async def test_loop_ends_normally_on_end_turn() -> None:
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="end_turn",
            blocks=[{"type": "text", "text": "all done"}],
        )
    )
    registry = MagicMock()
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_dispatches_non_terminal_tools_then_continues() -> None:
    """Search tool calls should still be dispatched and the loop continues."""
    from hypothesis_engine.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_search",
                        "name": "search",
                        "input": {"q": "foo"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "call_record",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )

    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={"ok": True},
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 2
    # search was dispatched; record_hypothesis was NOT (terminal)
    assert registry.call.await_count == 1
    names = [tc["name"] for tc in result.tool_calls]
    assert names == ["search", "record_hypothesis"]


@pytest.mark.asyncio
async def test_terminal_waits_for_required_supporting_tools() -> None:
    from hypothesis_engine.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_early",
                        "name": "record_hypothesis",
                        "input": {"title": "too early"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "capability_search",
                        "name": "capability_search",
                        "input": {"query": "graphene Raman"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_final",
                        "name": "record_hypothesis",
                        "input": {"title": "grounded"},
                    }
                ],
            ),
        ]
    )
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {
                "name": "capability_search",
                "description": "search catalog",
                "input_schema": {},
            },
            {
                "name": "record_hypothesis",
                "description": "finish",
                "input_schema": {},
            },
        ],
        max_output_tokens=512,
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(is_error=False, content={"results": []}))

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        terminal_required_tool_names=("capability_search",),
        terminal_required_tool_hint="Inspect the catalog before finalizing.",
    )

    assert result.iterations == 2
    assert client.call.await_count == 2
    assert registry.call.await_count == 1
    assert [call["name"] for call in result.tool_calls] == [
        "record_hypothesis",
        "capability_search",
        "record_hypothesis",
    ]
    assert result.tool_calls[0]["error"] == "blocked_terminal_requires_tools"
    assert result.tool_calls[2]["reused_blocked_terminal"] is True
    second_spec = client.call.await_args_list[1].args[0]
    rendered = str(second_spec.extra_messages)
    assert "capability_search" in rendered
    assert "Inspect the catalog before finalizing." in rendered


@pytest.mark.asyncio
async def test_invalid_capability_validation_does_not_satisfy_terminal_guard() -> None:
    from hypothesis_engine.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "validate_invalid",
                        "name": "capability_validate_workflow",
                        "input": {"study_plan": [{"component_id": "theory"}]},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_blocked",
                        "name": "record_hypothesis",
                        "input": {"title": "candidate", "study_plan": []},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "validate_repaired",
                        "name": "capability_validate_workflow",
                        "input": {"study_plan": [{"component_id": "theory"}]},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_final",
                        "name": "record_hypothesis",
                        "input": {"title": "repaired", "study_plan": []},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(
        side_effect=[
            ToolResult(content={"status": "invalid", "issues": [{"severity": "error"}]}),
            ToolResult(content={"status": "partial", "issues": []}),
        ]
    )
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="debate", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {
                "name": "capability_validate_workflow",
                "description": "validate",
                "input_schema": {},
            },
            {
                "name": "record_hypothesis",
                "description": "finish",
                "input_schema": {},
            },
        ],
        max_output_tokens=512,
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        terminal_required_tool_names=("capability_validate_workflow",),
    )

    assert result.iterations == 3
    assert client.call.await_count == 3
    assert registry.call.await_count == 2
    assert [call["name"] for call in result.tool_calls] == [
        "capability_validate_workflow",
        "record_hypothesis",
        "capability_validate_workflow",
        "record_hypothesis",
    ]
    assert result.tool_calls[1]["error"] == "blocked_terminal_requires_tools"
    assert result.tool_calls[0]["result"]["status"] == "invalid"
    assert result.tool_calls[2]["result"]["status"] == "partial"
    assert result.tool_calls[3]["reused_blocked_terminal"] is True


@pytest.mark.asyncio
async def test_end_turn_after_blocked_terminal_keeps_required_tool_loop_active() -> None:
    from hypothesis_engine.tools.base import ToolResult

    candidate = {"title": "preserve me", "study_plan": [{"component_id": "theory"}]}
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_early",
                        "name": "record_hypothesis",
                        "input": candidate,
                    }
                ],
            ),
            _fake_response(
                stop_reason="end_turn",
                blocks=[{"type": "text", "text": "done"}],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search",
                        "name": "capability_search",
                        "input": {"query": "TMD methods"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_final",
                        "name": "record_hypothesis",
                        "input": candidate,
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(content={"results": []}))
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="debate", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "capability_search", "description": "search", "input_schema": {}},
            {
                "name": "record_hypothesis",
                "description": "finish",
                "input_schema": {},
            },
        ],
        max_output_tokens=512,
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        terminal_required_tool_names=("capability_search",),
    )

    assert result.iterations == 3
    assert client.call.await_count == 3
    assert registry.call.await_count == 1
    third_call_spec = client.call.await_args_list[2].args[0]
    assert "preserve me" in str(third_call_spec.extra_messages)
    assert result.last_blocked_terminal_input == candidate
    assert result.tool_calls[-1]["reused_blocked_terminal"] is True


@pytest.mark.asyncio
async def test_ollama_repairs_invalid_blocked_terminal_after_required_tool() -> None:
    from hypothesis_engine.tools.base import ToolResult

    invalid_candidate = {"title": "needs repair"}
    repaired_candidate = {"title": "repaired", "statement": "complete"}
    client = MagicMock()
    client.supports_terminal_record_finalization = True
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "terminal_early",
                        "name": "record_hypothesis",
                        "input": invalid_candidate,
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search",
                        "name": "capability_search",
                        "input": {"query": "TMD methods"},
                    }
                ],
            ),
        ]
    )
    client.finalize_terminal_record = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "terminal_repaired",
                    "name": "record_hypothesis",
                    "input": repaired_candidate,
                }
            ],
        )
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(content={"results": []}))
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="debate", model="gpt-oss:120b"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "capability_search", "description": "search", "input_schema": {}},
            {
                "name": "record_hypothesis",
                "description": "finish",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "statement": {"type": "string"},
                    },
                    "required": ["title", "statement"],
                    "additionalProperties": False,
                },
            },
        ],
        max_output_tokens=512,
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        terminal_required_tool_names=("capability_search",),
    )

    assert result.iterations == 2
    client.finalize_terminal_record.assert_awaited_once()
    repair_kwargs = client.finalize_terminal_record.await_args.kwargs
    assert repair_kwargs["candidate_input"] == invalid_candidate
    assert any("statement" in item for item in repair_kwargs["validation_errors"])
    assert result.response.raw.content[0].input == repaired_candidate
    assert result.tool_calls[-1]["terminal_finalization"] is True
    assert result.tool_calls[-1]["terminal_finalization_mode"] == "repair"


@pytest.mark.asyncio
async def test_loop_exhaustion_synthesizes_terminal_without_cached_candidate() -> None:
    from hypothesis_engine.tools.base import ToolResult

    terminal = {"title": "Synthesized", "statement": "Complete record"}
    client = MagicMock()
    client.supports_terminal_record_finalization = True
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "search",
                    "name": "capability_search",
                    "input": {"query": "TMD methods"},
                }
            ],
        )
    )
    client.finalize_terminal_record = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "terminal",
                    "name": "record_hypothesis",
                    "input": terminal,
                }
            ],
        )
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(return_value=ToolResult(content={"results": []}))
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="gpt-oss:120b"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "capability_search", "description": "", "input_schema": {}},
            {
                "name": "record_hypothesis",
                "description": "",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "statement": {"type": "string"},
                    },
                    "required": ["title", "statement"],
                },
            },
        ],
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=2,
        force_terminal_tool="record_hypothesis",
    )

    assert result.iterations == 2
    assert client.call.await_count == 2
    client.finalize_terminal_record.assert_awaited_once()
    finalize_kwargs = client.finalize_terminal_record.await_args.kwargs
    assert finalize_kwargs["candidate_input"] is None
    assert len(finalize_kwargs["source_spec"].extra_messages) == 4
    assert result.response.raw.content[0].input == terminal
    assert result.tool_calls[-1]["terminal_finalization_mode"] == "synthesis"


@pytest.mark.asyncio
async def test_terminal_synthesis_repairs_first_invalid_finalization() -> None:
    invalid = {"title": "Incomplete"}
    valid = {"title": "Complete", "statement": "Now valid"}
    client = MagicMock()
    client.supports_terminal_record_finalization = True
    client.call = AsyncMock(
        return_value=_fake_response(stop_reason="end_turn", blocks=[{"type": "text"}])
    )
    client.finalize_terminal_record = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "invalid",
                        "name": "record_hypothesis",
                        "input": invalid,
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "valid",
                        "name": "record_hypothesis",
                        "input": valid,
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="gpt-oss:120b"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {
                "name": "record_hypothesis",
                "description": "",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "statement": {"type": "string"},
                    },
                    "required": ["title", "statement"],
                },
            }
        ],
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=8,
        force_terminal_tool="record_hypothesis",
    )

    assert result.iterations == 1
    assert client.finalize_terminal_record.await_count == 2
    second_kwargs = client.finalize_terminal_record.await_args_list[1].kwargs
    assert second_kwargs["candidate_input"] == invalid
    assert any("statement" in error for error in second_kwargs["validation_errors"])
    assert [attempt["terminal_finalization_mode"] for attempt in result.tool_calls] == [
        "synthesis",
        "repair",
    ]
    assert result.response.raw.content[0].input == valid


@pytest.mark.asyncio
async def test_unadvertised_tool_is_rejected_without_registry_dispatch() -> None:
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "hallucinated",
                        "name": "arxiv_search",
                        "input": {"query": "should not run"},
                    }
                ],
            ),
            _fake_response(stop_reason="end_turn", blocks=[{"type": "text", "text": "done"}]),
        ]
    )
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock()
    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="qwen3.6:35b"),
        user_blocks=[CachedBlock("go")],
        tools=[{"name": "capability_search", "description": "", "input_schema": {}}],
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=2,
    )

    registry.call.assert_not_awaited()
    assert result.tool_calls == [
        {
            "name": "arxiv_search",
            "args": {"query": "should not run"},
            "is_error": True,
            "duration_ms": 0,
            "error": "tool_not_in_active_schema",
        }
    ]
    followup = client.call.await_args_list[1].args[0]
    assert "tool_not_in_active_schema" in str(followup.extra_messages)


@pytest.mark.asyncio
async def test_loop_terminates_on_record_review() -> None:
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[
                {
                    "type": "tool_use",
                    "id": "c",
                    "name": "record_review",
                    "input": {"verdict": "accept"},
                }
            ],
        )
    )
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=MagicMock(),
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_loop_terminates_on_custom_terminal_tool() -> None:
    """The terminal-tool set is configurable via kwarg."""
    client = MagicMock()
    client.call = AsyncMock(
        return_value=_fake_response(
            stop_reason="tool_use",
            blocks=[{"type": "tool_use", "id": "c", "name": "my_done_signal", "input": {}}],
        )
    )
    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=MagicMock(),
        max_iters=8,
        parallel_cap=4,
        tool_timeout_s=1.0,
        terminal_tool_names=("my_done_signal",),
    )
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_force_terminal_tool_on_final_iteration() -> None:
    """A model that only ever searches should be forced to record on the last
    iteration when force_terminal_tool is set — instead of exhausting the loop."""
    from hypothesis_engine.tools.base import ToolResult

    # The model keeps searching forever unless tool_choice forces a specific
    # tool — exactly how a real provider behaves under a forced tool_choice.
    def _respect_tool_choice(spec, *_a, **_k):
        tc = spec.tool_choice or {}
        if tc.get("type") == "tool" and tc.get("name") == "record_hypothesis":
            return _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "r",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            )
        return _fake_response(
            stop_reason="tool_use",
            blocks=[{"type": "tool_use", "id": "s", "name": "search", "input": {"q": "x"}}],
        )

    client = MagicMock()
    client.call = AsyncMock(side_effect=_respect_tool_choice)
    registry = MagicMock()
    registry._cfg = SimpleNamespace()
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={"n": 0, "results": []},
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_spec(),
        ctx=_ctx(),
        registry=registry,
        max_iters=3,
        parallel_cap=4,
        tool_timeout_s=1.0,
        force_terminal_tool="record_hypothesis",
    )
    # The loop committed on the forced final iteration instead of exhausting.
    assert result.iterations == 3
    assert result.tool_calls[-1]["name"] == "record_hypothesis"
    # The final (3rd) call forced tool_choice to record_hypothesis.
    final_spec = client.call.await_args_list[-1].args[0]
    assert final_spec.tool_choice == {"type": "tool", "name": "record_hypothesis"}
    # Earlier calls used the original auto tool_choice (no forcing).
    first_spec = client.call.await_args_list[0].args[0]
    assert first_spec.tool_choice != {"type": "tool", "name": "record_hypothesis"}


def _rag_ready_cfg(tmp_path):
    root = tmp_path / "s" / "rag"
    root.mkdir(parents=True)
    (root / "kb.index").write_bytes(b"index")
    (root / "kb.pkl").write_bytes(b"meta")
    (root / "manifest.json").write_text(
        '{"papers":{"p1":{"indexed":true,"url":"https://arxiv.org/pdf/1234.5678"}}}'
    )

    class Cfg:
        rag = SimpleNamespace(enabled=True)

        def session_artifact_dir(self, session_id: str):
            return tmp_path / session_id

        def session_rag_dir(self, session_id: str):
            return tmp_path / session_id / "rag"

    return Cfg()


def _rag_spec() -> AgentCallSpec:
    return AgentCallSpec(
        route=ModelRoute(agent="reflection", mode="full", model="x"),
        user_blocks=[CachedBlock("review")],
        tools=[
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {"name": "rag_retrieve_context", "description": "", "input_schema": {}},
            {"name": "record_review", "description": "", "input_schema": {}},
        ],
        max_output_tokens=512,
    )


@pytest.mark.asyncio
async def test_rag_ready_session_blocks_search_until_rag_retrieval(tmp_path) -> None:
    from hypothesis_engine.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search_1",
                        "name": "arxiv_search",
                        "input": {"query": "defect formation"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "rag_1",
                        "name": "rag_retrieve_context",
                        "input": {"query": "defect formation"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_1",
                        "name": "record_review",
                        "input": {"verdict": "neutral", "evidence": []},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = _rag_ready_cfg(tmp_path)
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={
                "rerank_chunks": [{"url": "https://arxiv.org/pdf/1234.5678", "text": "context"}]
            },
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_rag_spec(),
        ctx=CallContext(session_id="s", task_id="t", agent="reflection", action="review"),
        registry=registry,
        max_iters=5,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )

    assert registry.call.await_count == 1
    assert registry.call.await_args_list[0].args[0] == "rag_retrieve_context"
    assert result.seen_urls == {"https://arxiv.org/pdf/1234.5678"}
    assert any(
        tc.get("error") == "blocked_search_requires_rag_retrieve_context"
        for tc in result.tool_calls
    )
    assert [tc["name"] for tc in result.tool_calls] == [
        "arxiv_search",
        "rag_retrieve_context",
        "record_review",
    ]


@pytest.mark.asyncio
async def test_rag_ready_session_blocks_record_review_until_rag_retrieval(tmp_path) -> None:
    from hypothesis_engine.tools.base import ToolResult

    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_early",
                        "name": "record_review",
                        "input": {"verdict": "neutral", "evidence": []},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "rag_1",
                        "name": "rag_retrieve_context",
                        "input": {"query": "defect formation"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_final",
                        "name": "record_review",
                        "input": {"verdict": "neutral", "evidence": []},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = _rag_ready_cfg(tmp_path)
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={
                "rerank_chunks": [{"url": "https://arxiv.org/pdf/1234.5678", "text": "context"}]
            },
            duration_ms=1,
        )
    )

    result = await run_tool_loop(
        client,
        spec=_rag_spec(),
        ctx=CallContext(session_id="s", task_id="t", agent="reflection", action="review"),
        registry=registry,
        max_iters=5,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )

    assert registry.call.await_count == 1
    assert registry.call.await_args_list[0].args[0] == "rag_retrieve_context"
    assert any(
        tc.get("error") == "blocked_terminal_requires_rag_retrieve_context"
        for tc in result.tool_calls
    )
    assert [tc["name"] for tc in result.tool_calls] == [
        "record_review",
        "rag_retrieve_context",
        "record_review",
    ]


def test_arxiv_source_key_collapses_versions_and_abs_pdf_urls() -> None:
    assert _source_key("https://arxiv.org/pdf/2109.11880v1") == "arxiv:2109.11880"
    assert _source_key("https://arxiv.org/abs/2109.11880v2") == "arxiv:2109.11880"
    assert _source_key("https://arxiv.org/pdf/2109.11880.pdf") == "arxiv:2109.11880"


@pytest.mark.asyncio
async def test_fetch_guard_blocks_session_duplicate_when_new_candidates_exist(tmp_path) -> None:
    from hypothesis_engine.tools.base import ToolResult

    old_url = "https://arxiv.org/pdf/2109.11880v1"
    new_url = "https://arxiv.org/pdf/2503.22476v2"
    papers_dir = tmp_path / "s" / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "old.json").write_text(
        '{"url":"https://arxiv.org/abs/2109.11880v3","text":"previous full text"}'
    )

    class Cfg:
        def session_artifact_dir(self, session_id: str):
            return tmp_path / session_id

    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {"name": "web_fetch", "description": "", "input_schema": {}},
        ],
        max_output_tokens=512,
    )
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search_1",
                        "name": "arxiv_search",
                        "input": {"query": "x"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "fetch_old",
                        "name": "web_fetch",
                        "input": {"url": old_url},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "fetch_new",
                        "name": "web_fetch",
                        "input": {"url": new_url},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = Cfg()
    registry.call = AsyncMock(
        side_effect=[
            ToolResult(
                is_error=False,
                content={"results": [{"pdf_url": old_url}, {"pdf_url": new_url}]},
                duration_ms=1,
            ),
            ToolResult(
                is_error=False,
                content={"url": new_url, "requested_url": new_url, "text": "new full text"},
                duration_ms=1,
            ),
        ]
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=6,
        parallel_cap=4,
        tool_timeout_s=1.0,
        terminal_min_seen_urls=1,
    )

    assert registry.call.await_count == 2
    assert [call.args[0] for call in registry.call.await_args_list] == ["arxiv_search", "web_fetch"]
    assert registry.call.await_args_list[-1].args[1]["url"] == new_url
    assert any(tc.get("error") == "blocked_duplicate_web_fetch" for tc in result.tool_calls)
    assert result.seen_urls == {new_url}


@pytest.mark.asyncio
async def test_fetch_guard_blocks_direct_session_duplicate_and_requests_search(tmp_path) -> None:
    from hypothesis_engine.tools.base import ToolResult

    old_url = "https://arxiv.org/pdf/2109.11880v1"
    new_url = "https://arxiv.org/pdf/2503.22476v2"
    papers_dir = tmp_path / "s" / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "old.json").write_text(
        '{"url":"https://arxiv.org/pdf/2109.11880v1","text":"previous full text"}'
    )

    class Cfg:
        def session_artifact_dir(self, session_id: str):
            return tmp_path / session_id

    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {"name": "web_fetch", "description": "", "input_schema": {}},
        ],
        max_output_tokens=512,
    )
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "fetch_old",
                        "name": "web_fetch",
                        "input": {"url": old_url},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search_1",
                        "name": "arxiv_search",
                        "input": {"query": "x"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "fetch_new",
                        "name": "web_fetch",
                        "input": {"url": new_url},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = Cfg()
    registry.call = AsyncMock(
        side_effect=[
            ToolResult(
                is_error=False,
                content={"results": [{"pdf_url": old_url}, {"pdf_url": new_url}]},
                duration_ms=1,
            ),
            ToolResult(
                is_error=False,
                content={"url": new_url, "requested_url": new_url, "text": "new full text"},
                duration_ms=1,
            ),
        ]
    )

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=6,
        parallel_cap=4,
        tool_timeout_s=1.0,
        terminal_min_seen_urls=1,
    )

    assert registry.call.await_count == 2
    assert [call.args[0] for call in registry.call.await_args_list] == ["arxiv_search", "web_fetch"]
    assert registry.call.await_args_list[-1].args[1]["url"] == new_url
    assert any(tc.get("error") == "blocked_duplicate_web_fetch" for tc in result.tool_calls)
    assert result.seen_urls == {new_url}


@pytest.mark.asyncio
async def test_fetch_guard_never_dispatches_session_duplicate_when_discovery_exists(
    tmp_path,
) -> None:
    old_url = "https://arxiv.org/pdf/2109.11880v1"
    papers_dir = tmp_path / "s" / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "old.json").write_text(
        '{"url":"https://arxiv.org/abs/2109.11880v3","text":"previous full text"}'
    )

    class Cfg:
        def session_artifact_dir(self, session_id: str):
            return tmp_path / session_id

    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {"name": "web_fetch", "description": "", "input_schema": {}},
        ],
        max_output_tokens=512,
    )
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "fetch_old",
                        "name": "web_fetch",
                        "input": {"url": old_url},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = Cfg()

    result = await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=6,
        parallel_cap=4,
        tool_timeout_s=1.0,
        terminal_min_seen_urls=0,
    )

    registry.call.assert_not_called()
    assert any(tc.get("error") == "blocked_duplicate_web_fetch" for tc in result.tool_calls)
    assert result.tool_calls[-1]["name"] == "record_hypothesis"


@pytest.mark.asyncio
async def test_search_results_hide_session_duplicate_urls_from_model_context(
    tmp_path,
) -> None:
    from hypothesis_engine.tools.base import ToolResult

    old_url = "https://arxiv.org/pdf/2109.11880v1"
    new_url = "https://arxiv.org/pdf/2503.22476v2"
    papers_dir = tmp_path / "s" / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / "old.json").write_text(
        '{"url":"https://arxiv.org/abs/2109.11880v3","text":"previous full text"}'
    )

    class Cfg:
        def session_artifact_dir(self, session_id: str):
            return tmp_path / session_id

    spec = AgentCallSpec(
        route=ModelRoute(agent="generation", mode="literature", model="x"),
        user_blocks=[CachedBlock("go")],
        tools=[
            {"name": "arxiv_search", "description": "", "input_schema": {}},
            {"name": "web_fetch", "description": "", "input_schema": {}},
        ],
        max_output_tokens=512,
    )
    client = MagicMock()
    client.call = AsyncMock(
        side_effect=[
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "search_1",
                        "name": "arxiv_search",
                        "input": {"query": "x"},
                    }
                ],
            ),
            _fake_response(
                stop_reason="tool_use",
                blocks=[
                    {
                        "type": "tool_use",
                        "id": "record_1",
                        "name": "record_hypothesis",
                        "input": {"title": "t", "statement": "s"},
                    }
                ],
            ),
        ]
    )
    registry = MagicMock()
    registry._cfg = Cfg()
    registry.call = AsyncMock(
        return_value=ToolResult(
            is_error=False,
            content={
                "results": [
                    {"title": "old", "pdf_url": old_url, "abs_url": old_url},
                    {"title": "new", "pdf_url": new_url, "abs_url": new_url},
                ]
            },
            duration_ms=1,
        )
    )

    await run_tool_loop(
        client,
        spec=spec,
        ctx=_ctx(),
        registry=registry,
        max_iters=4,
        parallel_cap=4,
        tool_timeout_s=1.0,
    )

    next_spec = client.call.await_args_list[1].args[0]
    next_context = str(next_spec.extra_messages)
    assert old_url not in next_context
    assert new_url in next_context
    assert "session_duplicate_results_omitted" in next_context


def test_seen_urls_ignore_search_discovery_metadata() -> None:
    result = {
        "is_error": False,
        "content": {
            "results": [
                {
                    "pdf_url": "https://arxiv.org/pdf/1234.5678",
                    "abs_url": "https://arxiv.org/abs/1234.5678",
                }
            ]
        },
    }
    assert _recordable_seen_urls("arxiv_search", {"query": "x"}, result) == []


def test_seen_urls_record_successful_web_fetch_text() -> None:
    result = {
        "is_error": False,
        "content": {"url": "https://example.test/paper", "text": "full extracted text"},
    }
    assert _recordable_seen_urls("web_fetch", {"url": "https://example.test/paper"}, result) == [
        "https://example.test/paper"
    ]


def test_seen_urls_ignore_failed_or_empty_web_fetch() -> None:
    assert (
        _recordable_seen_urls(
            "web_fetch",
            {"url": "https://example.test/paper"},
            {"is_error": True, "content": {"url": "https://example.test/paper", "text": "body"}},
        )
        == []
    )
    assert (
        _recordable_seen_urls(
            "web_fetch",
            {"url": "https://example.test/paper"},
            {"is_error": False, "content": {"url": "https://example.test/paper", "text": ""}},
        )
        == []
    )


def test_seen_urls_record_successful_web_fetch_snippet_even_when_truncated() -> None:
    result = {
        "is_error": False,
        "content": {
            "url": "https://example.test/paper",
            "text": "partial text",
            "truncated": True,
        },
    }
    assert _recordable_seen_urls("web_fetch", {"url": "https://example.test/paper"}, result) == [
        "https://example.test/paper"
    ]


def test_web_fetch_tool_result_is_truncated_for_context() -> None:
    tool_use = SimpleNamespace(id="fetch_1", name="web_fetch")
    block = _tool_result_block(
        tool_use,
        {
            "is_error": False,
            "content": {"url": "https://example.test/paper", "text": "x" * 70_000},
        },
    )
    assert block["type"] == "tool_result"
    assert len(block["content"]) < 12_000
    assert "truncated_for_context" in block["content"]
    assert "source_context_mode" in block["content"]
    assert "compact_excerpt" in block["content"]
    assert "full_text_available_in_artifact_cache" in block["content"]


def test_search_results_compacted_for_model_context_preserves_full_summaries() -> None:
    final_claim_summary = (
        "Opening background. "
        + "mechanistic detail " * 120
        + "Important final claim: substrate protection changes the damage threshold."
    )
    records = [
        {
            "title": f"Paper {i}",
            "authors": [f"First {i}", f"Second {i}", f"Third {i}"],
            "summary": final_claim_summary,
            "pdf_url": f"https://arxiv.org/pdf/2501.{i:05d}v1",
            "abs_url": f"https://arxiv.org/abs/2501.{i:05d}v1",
        }
        for i in range(35)
    ]
    result = {
        "is_error": False,
        "content": {
            "query": "low energy ion implantation monolayer TMD damage",
            "n": len(records),
            "filtered_out": 7,
            "results": records,
            "rag_ingest": {"downloaded": 5, "ingested": 5},
        },
        "duration_ms": 12,
    }

    compacted = _result_for_model_context(
        tool_name="arxiv_search",
        result=result,
        session_fetched_source_keys=set(),
    )

    content = compacted["content"]
    assert len(content["results"]) == SEARCH_RESULTS_CONTEXT_LIMIT
    assert content["n"] == SEARCH_RESULTS_CONTEXT_LIMIT
    assert content["total_results"] == 35
    assert content["model_context_results_omitted"] == 5
    assert content["rag_ingest"] == {"downloaded": 5, "ingested": 5}
    assert content["results"][0]["authors"] == ["First 0"]
    assert content["results"][0]["authors_omitted"] == 2
    assert content["results"][0]["summary"].endswith(
        "Important final claim: substrate protection changes the damage threshold."
    )

    assert len(result["content"]["results"]) == 35
    assert result["content"]["results"][0]["authors"] == ["First 0", "Second 0", "Third 0"]
