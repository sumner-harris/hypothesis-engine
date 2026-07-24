# Modified from the original work.
"""The assistant↔tool_use↔tool_result loop.

The agent gives us:
- An initial AgentCallSpec (system + user blocks, tools, tool_choice).
- A ToolRegistry (or just the subset relevant for this agent).
- A max_iters cap.

We drive turns until the model returns a non-tool-use stop_reason, or we hit
the cap (which surfaces as ToolLoopExhausted to the calling agent).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from ..ids import tool_run_id
from ..tools.base import ToolCtx
from ..tools.registry import ToolRegistry
from .anthropic_client import AgentCallSpec, AnthropicClient, AnthropicResponse, CallContext

WEB_FETCH_CONTEXT_TEXT_CHARS = 6_000
SEARCH_RESULTS_CONTEXT_LIMIT = 30


class ToolLoopExhausted(RuntimeError):
    def __init__(self, agent: str, iters: int):
        super().__init__(f"tool loop for agent {agent!r} exhausted after {iters} iterations")
        self.agent = agent
        self.iters = iters


@dataclass
class ToolLoopResult:
    response: AnthropicResponse  # final assistant message
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    citation_candidates: list[dict[str, Any]] = field(default_factory=list)
    last_blocked_terminal_input: dict[str, Any] | None = None
    """URLs whose content was successfully previewed during the loop.

    Used by structured-output validation to reject hallucinated citations:
    Generation's record_hypothesis.citations[].url must be in this set.
    Search-discovery metadata alone is not enough; the agent must call
    web_fetch and receive usable text for a source to become citeable. The
    text may be a compact excerpt; full paper text is not required in-context.
    """


@dataclass
class _TerminalFinalizationResult:
    response: AnthropicResponse | None
    candidate_input: dict[str, Any] | None
    validation_errors: list[str]
    attempts: list[dict[str, Any]]


async def run_tool_loop(
    client: AnthropicClient,
    *,
    spec: AgentCallSpec,
    ctx: CallContext,
    registry: ToolRegistry,
    max_iters: int,
    parallel_cap: int = 4,
    tool_timeout_s: float = 30.0,
    force_terminal_tool: str | None = None,
    terminal_tool_names: tuple[str, ...] = (
        "record_hypothesis",
        "record_review",
        "record_system_feedback",
        "record_research_plan",
    ),
    terminal_min_seen_urls: int = 0,
    terminal_requirement_hint: str | None = None,
    terminal_required_tool_names: tuple[str, ...] = (),
    terminal_required_tool_hint: str | None = None,
) -> ToolLoopResult:
    """Drive the assistant ↔ tool_use ↔ tool_result loop.

    Loop termination:
    - stop_reason != "tool_use" — the model signalled end_turn.
    - The assistant response contains a `terminal_tool_names` call. These are
      virtual "structured output capture" tools (e.g. `record_hypothesis`):
      the assistant has already produced its final answer in tool_use.input,
      so dispatching the tool is unnecessary and we should not invite the
      model to call it again. Claude reliably ends its turn after calling
      these; Gemini / OpenAI-compat models do not, so we short-circuit
      explicitly. Without this short-circuit the loop will repeatedly
      re-invite the recording tool until max_iters and then raise
      ToolLoopExhausted — even though a perfectly good record was emitted
      on the first call.
    - max_iters reached — raise ToolLoopExhausted.

    `force_terminal_tool`: if set, providers with a dedicated finalization
    path receive a fresh schema-constrained synthesis/repair call after normal
    exploration (or after an early end_turn). Other providers retain the
    legacy behavior of forcing the tool on the final allowed iteration.
    """
    seen_urls: set[str] = set()
    session_fetched_source_keys = _session_fetched_source_keys(registry._cfg, ctx.session_id)
    fetchable_urls: list[str] = []
    web_fetch_attempts = 0
    fetch_guard_blocks = 0
    duplicate_web_fetch_blocks = 0
    rag_first_guard_blocks = 0
    rag_retrieve_attempts = 0
    terminal_required_tool_guard_blocks = 0
    successful_tool_names: set[str] = set()
    last_blocked_terminal_input: dict[str, Any] | None = None
    last_blocked_terminal_name: str | None = None
    tool_calls_log: list[dict[str, Any]] = []
    citation_candidates: list[dict[str, Any]] = []
    citation_candidate_urls: set[str] = set()
    iterations = 0
    current_spec = spec
    terminal_set = set(terminal_tool_names)
    supports_terminal_finalization = (
        getattr(client, "supports_terminal_record_finalization", False) is True
    )

    last: AnthropicResponse | None = None

    while iterations < max_iters:
        iterations += 1
        # On the final allowed iteration, optionally force the recording tool so
        # the model commits instead of burning its last turn on another search.
        call_spec = current_spec
        if (
            force_terminal_tool
            and iterations == max_iters
            and not supports_terminal_finalization
            and not _needs_more_seen_urls(
                min_seen_urls=terminal_min_seen_urls,
                seen_urls=seen_urls,
                fetchable_urls=fetchable_urls,
                session_fetched_source_keys=session_fetched_source_keys,
                tools=current_spec.tools,
                web_fetch_attempts=web_fetch_attempts,
                guard_blocks=fetch_guard_blocks,
            )
        ):
            call_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice={"type": "tool", "name": force_terminal_tool},
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=current_spec.extra_messages,
                reasoning_effort=current_spec.reasoning_effort,
            )
        resp = await client.call(call_spec, ctx)
        last = resp
        stop = getattr(resp.raw, "stop_reason", None)

        if stop != "tool_use":
            available_tool_names = _tool_names(current_spec.tools)
            missing_required_tools = [
                name
                for name in terminal_required_tool_names
                if name in available_tool_names and name not in successful_tool_names
            ]
            if (
                last_blocked_terminal_input is not None
                and missing_required_tools
                and terminal_required_tool_guard_blocks < 3
            ):
                assistant_blocks = _content_to_dicts(resp.raw.content)
                next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
                if assistant_blocks:
                    next_messages.append({"role": "assistant", "content": assistant_blocks})
                next_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _terminal_required_tools_message(
                                    terminal_name=force_terminal_tool or "terminal record",
                                    missing_tool_names=missing_required_tools,
                                    hint=terminal_required_tool_hint,
                                    candidate_input=last_blocked_terminal_input,
                                ),
                            }
                        ],
                    }
                )
                current_spec = AgentCallSpec(
                    route=current_spec.route,
                    system_blocks=current_spec.system_blocks,
                    user_blocks=current_spec.user_blocks,
                    tools=current_spec.tools,
                    tool_choice=current_spec.tool_choice,
                    max_output_tokens=current_spec.max_output_tokens,
                    stop_sequences=current_spec.stop_sequences,
                    extra_messages=next_messages,
                    reasoning_effort=current_spec.reasoning_effort,
                )
                terminal_required_tool_guard_blocks += 1
                continue
            if (
                force_terminal_tool
                and supports_terminal_finalization
                and not missing_required_tools
            ):
                break
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
                citation_candidates=citation_candidates,
                last_blocked_terminal_input=last_blocked_terminal_input,
            )

        # Extract tool_use blocks from the assistant response
        tool_uses = [b for b in resp.raw.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            missing_required_tools = [
                name
                for name in terminal_required_tool_names
                if name in _tool_names(current_spec.tools)
                and name not in successful_tool_names
            ]
            if (
                force_terminal_tool
                and supports_terminal_finalization
                and not missing_required_tools
            ):
                break
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
                citation_candidates=citation_candidates,
                last_blocked_terminal_input=last_blocked_terminal_input,
            )

        available_tool_names = _tool_names(current_spec.tools)
        allowed_response_tool_names = available_tool_names | terminal_set
        unexpected_tool_uses = [
            block
            for block in tool_uses
            if str(getattr(block, "name", "") or "") not in allowed_response_tool_names
        ]
        if unexpected_tool_uses:
            # Never dispatch a hallucinated or disabled tool merely because a
            # provider returned a syntactically valid tool call. Reject the
            # entire response so every tool_use receives a paired error result.
            context_results: list[dict[str, Any]] = []
            for block in tool_uses:
                name = str(getattr(block, "name", "") or "")
                if name in allowed_response_tool_names:
                    error = "batch_rejected_due_unadvertised_tool"
                else:
                    error = "tool_not_in_active_schema"
                result = {
                    "is_error": True,
                    "content": {
                        "error": error,
                        "tool": name,
                        "active_tools": sorted(available_tool_names),
                    },
                    "duration_ms": 0,
                }
                context_results.append(result)
                tool_calls_log.append(
                    {
                        "name": name,
                        "args": dict(getattr(block, "input", {}) or {}),
                        "is_error": True,
                        "duration_ms": 0,
                        "error": error,
                    }
                )
            next_messages = list(current_spec.extra_messages)
            next_messages.append(
                {"role": "assistant", "content": _content_to_dicts(resp.raw.content)}
            )
            next_messages.append(
                {
                    "role": "user",
                    "content": [
                        _tool_result_block(tool_use, result)
                        for tool_use, result in zip(
                            tool_uses, context_results, strict=True
                        )
                    ],
                }
            )
            current_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice=current_spec.tool_choice,
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=next_messages,
                reasoning_effort=current_spec.reasoning_effort,
            )
            continue

        terminal_uses = [b for b in tool_uses if getattr(b, "name", "") in terminal_set]
        if terminal_uses and _rag_retrieval_required(
            cfg=registry._cfg,
            session_id=ctx.session_id,
            tools=current_spec.tools,
            rag_retrieve_attempts=rag_retrieve_attempts,
            guard_blocks=rag_first_guard_blocks,
        ):
            for b in terminal_uses:
                tool_calls_log.append(
                    {
                        "name": getattr(b, "name", ""),
                        "args": dict(getattr(b, "input", {}) or {}),
                        "is_error": True,
                        "duration_ms": 0,
                        "error": "blocked_terminal_requires_rag_retrieve_context",
                    }
                )
            assistant_blocks = [
                b for b in _content_to_dicts(resp.raw.content) if b.get("type") != "tool_use"
            ]
            next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
            if assistant_blocks:
                next_messages.append({"role": "assistant", "content": assistant_blocks})
            next_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _rag_retrieval_required_message(
                                attempted_tool=terminal_uses[0].name,
                                terminal=True,
                            ),
                        }
                    ],
                }
            )
            current_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice=current_spec.tool_choice,
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=next_messages,
                reasoning_effort=current_spec.reasoning_effort,
            )
            rag_first_guard_blocks += 1
            continue

        missing_required_tools = [
            name
            for name in terminal_required_tool_names
            if name in available_tool_names and name not in successful_tool_names
        ]
        if terminal_uses and missing_required_tools and terminal_required_tool_guard_blocks < 3:
            last_blocked_terminal_input = dict(getattr(terminal_uses[0], "input", {}) or {})
            last_blocked_terminal_name = str(getattr(terminal_uses[0], "name", "") or "")
            for b in terminal_uses:
                tool_calls_log.append(
                    {
                        "name": getattr(b, "name", ""),
                        "args": dict(getattr(b, "input", {}) or {}),
                        "is_error": True,
                        "duration_ms": 0,
                        "error": "blocked_terminal_requires_tools",
                    }
                )
            assistant_blocks = [
                b for b in _content_to_dicts(resp.raw.content) if b.get("type") != "tool_use"
            ]
            next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
            if assistant_blocks:
                next_messages.append({"role": "assistant", "content": assistant_blocks})
            next_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _terminal_required_tools_message(
                                terminal_name=terminal_uses[0].name,
                                missing_tool_names=missing_required_tools,
                                hint=terminal_required_tool_hint,
                                candidate_input=last_blocked_terminal_input,
                            ),
                        }
                    ],
                }
            )
            current_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice=current_spec.tool_choice,
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=next_messages,
                reasoning_effort=current_spec.reasoning_effort,
            )
            terminal_required_tool_guard_blocks += 1
            continue

        # Early termination: if any tool_use is a terminal recording tool,
        # treat this response as the final assistant message. We still log
        # the call so observability sees it, but we do NOT dispatch (the
        # registry would return "unknown tool" anyway) unless the caller has
        # opted into a minimum fetched-source requirement that is still unmet.
        if terminal_uses:
            terminal_name = getattr(terminal_uses[0], "name", "")
            if _needs_more_seen_urls(
                min_seen_urls=terminal_min_seen_urls,
                seen_urls=seen_urls,
                fetchable_urls=fetchable_urls,
                session_fetched_source_keys=session_fetched_source_keys,
                tools=current_spec.tools,
                web_fetch_attempts=web_fetch_attempts,
                guard_blocks=fetch_guard_blocks,
            ):
                for b in terminal_uses:
                    tool_calls_log.append(
                        {
                            "name": getattr(b, "name", ""),
                            "args": dict(getattr(b, "input", {}) or {}),
                            "is_error": True,
                            "duration_ms": 0,
                            "error": "blocked_terminal_requires_web_fetch",
                        }
                    )
                assistant_blocks = [
                    b for b in _content_to_dicts(resp.raw.content) if b.get("type") != "tool_use"
                ]
                next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
                if assistant_blocks:
                    next_messages.append({"role": "assistant", "content": assistant_blocks})
                next_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _terminal_fetch_guard_message(
                                    terminal_name=terminal_name,
                                    min_seen_urls=terminal_min_seen_urls,
                                    seen_urls=seen_urls,
                                    fetchable_urls=fetchable_urls,
                                    session_fetched_source_keys=session_fetched_source_keys,
                                    hint=terminal_requirement_hint,
                                ),
                            }
                        ],
                    }
                )
                current_spec = AgentCallSpec(
                    route=current_spec.route,
                    system_blocks=current_spec.system_blocks,
                    user_blocks=current_spec.user_blocks,
                    tools=current_spec.tools,
                    tool_choice=current_spec.tool_choice,
                    max_output_tokens=current_spec.max_output_tokens,
                    stop_sequences=current_spec.stop_sequences,
                    extra_messages=next_messages,
                    reasoning_effort=current_spec.reasoning_effort,
                )
                fetch_guard_blocks += 1
                continue

            terminal_candidate = dict(
                getattr(terminal_uses[0], "input", {}) or {}
            )
            terminal_errors = (
                _terminal_schema_validation_errors(
                    tools=current_spec.tools,
                    terminal_name=terminal_name,
                    candidate_input=terminal_candidate,
                )
                if terminal_name in available_tool_names
                else []
            )
            if terminal_errors:
                last_blocked_terminal_input = terminal_candidate
                last_blocked_terminal_name = terminal_name
                tool_calls_log.append(
                    {
                        "name": terminal_name,
                        "args": terminal_candidate,
                        "is_error": True,
                        "duration_ms": 0,
                        "error": "invalid_terminal_schema",
                        "validation_errors": terminal_errors,
                    }
                )
                finalized = await _try_terminal_finalization(
                    client,
                    source_spec=current_spec,
                    ctx=ctx,
                    terminal_name=terminal_name,
                    candidate_input=terminal_candidate,
                    validation_errors=terminal_errors,
                )
                tool_calls_log.extend(finalized.attempts)
                if finalized.response is not None:
                    return ToolLoopResult(
                        response=finalized.response,
                        iterations=iterations,
                        tool_calls=tool_calls_log,
                        seen_urls=seen_urls,
                        citation_candidates=citation_candidates,
                        last_blocked_terminal_input=terminal_candidate,
                    )

                next_messages = list(current_spec.extra_messages)
                assistant_blocks = [
                    block
                    for block in _content_to_dicts(resp.raw.content)
                    if block.get("type") != "tool_use"
                ]
                if assistant_blocks:
                    next_messages.append(
                        {"role": "assistant", "content": assistant_blocks}
                    )
                next_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"The {terminal_name} arguments did not validate. "
                                    "Correct these schema errors before finalizing:\n"
                                    + "\n".join(finalized.validation_errors or terminal_errors)
                                ),
                            }
                        ],
                    }
                )
                current_spec = AgentCallSpec(
                    route=current_spec.route,
                    system_blocks=current_spec.system_blocks,
                    user_blocks=current_spec.user_blocks,
                    tools=current_spec.tools,
                    tool_choice=current_spec.tool_choice,
                    max_output_tokens=current_spec.max_output_tokens,
                    stop_sequences=current_spec.stop_sequences,
                    extra_messages=next_messages,
                    reasoning_effort=current_spec.reasoning_effort,
                )
                continue

            for b in tool_uses:
                tool_calls_log.append(
                    {
                        "name": getattr(b, "name", ""),
                        "args": dict(getattr(b, "input", {}) or {}),
                        "is_error": False,
                        "duration_ms": 0,
                    }
                )
            return ToolLoopResult(
                response=resp,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
                citation_candidates=citation_candidates,
                last_blocked_terminal_input=last_blocked_terminal_input,
            )

        rag_blocked_uses = _rag_first_blocked_discovery_uses(
            tool_uses,
            cfg=registry._cfg,
            session_id=ctx.session_id,
            tools=current_spec.tools,
            rag_retrieve_attempts=rag_retrieve_attempts,
            guard_blocks=rag_first_guard_blocks,
        )
        if rag_blocked_uses:
            for b in rag_blocked_uses:
                tool_calls_log.append(
                    {
                        "name": getattr(b, "name", ""),
                        "args": dict(getattr(b, "input", {}) or {}),
                        "is_error": True,
                        "duration_ms": 0,
                        "error": "blocked_search_requires_rag_retrieve_context",
                    }
                )
            assistant_blocks = [
                b for b in _content_to_dicts(resp.raw.content) if b.get("type") != "tool_use"
            ]
            next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
            if assistant_blocks:
                next_messages.append({"role": "assistant", "content": assistant_blocks})
            next_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _rag_retrieval_required_message(
                                attempted_tool=getattr(rag_blocked_uses[0], "name", "search"),
                                terminal=False,
                            ),
                        }
                    ],
                }
            )
            current_spec = AgentCallSpec(
                route=current_spec.route,
                system_blocks=current_spec.system_blocks,
                user_blocks=current_spec.user_blocks,
                tools=current_spec.tools,
                tool_choice=current_spec.tool_choice,
                max_output_tokens=current_spec.max_output_tokens,
                stop_sequences=current_spec.stop_sequences,
                extra_messages=next_messages,
                reasoning_effort=current_spec.reasoning_effort,
            )
            rag_first_guard_blocks += 1
            continue

        if _has_discovery_tool(current_spec.tools):
            duplicate_fetch_uses = _session_duplicate_web_fetch_uses(
                tool_uses,
                session_fetched_source_keys,
            )
            if duplicate_fetch_uses:
                for b in duplicate_fetch_uses:
                    tool_calls_log.append(
                        {
                            "name": getattr(b, "name", ""),
                            "args": dict(getattr(b, "input", {}) or {}),
                            "is_error": True,
                            "duration_ms": 0,
                            "error": "blocked_duplicate_web_fetch",
                        }
                    )
                duplicate_ids = {getattr(b, "id", None) for b in duplicate_fetch_uses}
                filtered_tool_uses = [
                    b for b in tool_uses if getattr(b, "id", None) not in duplicate_ids
                ]
                duplicate_web_fetch_blocks += 1
                if filtered_tool_uses:
                    tool_uses = filtered_tool_uses
                else:
                    assistant_blocks = [
                        b
                        for b in _content_to_dicts(resp.raw.content)
                        if b.get("type") != "tool_use"
                    ]
                    next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
                    if assistant_blocks:
                        next_messages.append({"role": "assistant", "content": assistant_blocks})
                    next_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _session_new_fetch_required_message(
                                        min_seen_urls=terminal_min_seen_urls,
                                        seen_urls=seen_urls,
                                        fetchable_urls=fetchable_urls,
                                        session_fetched_source_keys=session_fetched_source_keys,
                                        hint=terminal_requirement_hint,
                                    ),
                                }
                            ],
                        }
                    )
                    current_spec = AgentCallSpec(
                        route=current_spec.route,
                        system_blocks=current_spec.system_blocks,
                        user_blocks=current_spec.user_blocks,
                        tools=current_spec.tools,
                        tool_choice=current_spec.tool_choice,
                        max_output_tokens=current_spec.max_output_tokens,
                        stop_sequences=current_spec.stop_sequences,
                        extra_messages=next_messages,
                        reasoning_effort=current_spec.reasoning_effort,
                    )
                    fetch_guard_blocks += 1
                    continue

        if _needs_more_seen_urls(
            min_seen_urls=terminal_min_seen_urls,
            seen_urls=seen_urls,
            fetchable_urls=fetchable_urls,
            session_fetched_source_keys=session_fetched_source_keys,
            tools=current_spec.tools,
            web_fetch_attempts=web_fetch_attempts,
            guard_blocks=fetch_guard_blocks,
        ):
            web_fetch_uses = [tu for tu in tool_uses if getattr(tu, "name", "") == "web_fetch"]
            if web_fetch_uses:
                session_new_candidates = _session_new_fetchable_urls(
                    fetchable_urls,
                    seen_urls,
                    session_fetched_source_keys,
                )
                require_session_new = bool(session_new_candidates)
                preferred_web_fetch_uses = [
                    tu
                    for tu in web_fetch_uses
                    if _is_preferred_web_fetch(
                        tu,
                        seen_urls=seen_urls,
                        session_fetched_source_keys=session_fetched_source_keys,
                        require_session_new=require_session_new,
                    )
                ]
                if preferred_web_fetch_uses:
                    tool_uses = preferred_web_fetch_uses
                else:
                    for b in web_fetch_uses:
                        tool_calls_log.append(
                            {
                                "name": getattr(b, "name", ""),
                                "args": dict(getattr(b, "input", {}) or {}),
                                "is_error": True,
                                "duration_ms": 0,
                                "error": "blocked_duplicate_web_fetch",
                            }
                        )
                    assistant_blocks = [
                        b
                        for b in _content_to_dicts(resp.raw.content)
                        if b.get("type") != "tool_use"
                    ]
                    next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
                    if assistant_blocks:
                        next_messages.append({"role": "assistant", "content": assistant_blocks})
                    next_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": _web_fetch_required_message(
                                        min_seen_urls=terminal_min_seen_urls,
                                        seen_urls=seen_urls,
                                        fetchable_urls=fetchable_urls,
                                        session_fetched_source_keys=session_fetched_source_keys,
                                        hint=terminal_requirement_hint,
                                    ),
                                }
                            ],
                        }
                    )
                    current_spec = AgentCallSpec(
                        route=current_spec.route,
                        system_blocks=current_spec.system_blocks,
                        user_blocks=current_spec.user_blocks,
                        tools=current_spec.tools,
                        tool_choice=current_spec.tool_choice,
                        max_output_tokens=current_spec.max_output_tokens,
                        stop_sequences=current_spec.stop_sequences,
                        extra_messages=next_messages,
                        reasoning_effort=current_spec.reasoning_effort,
                    )
                    fetch_guard_blocks += 1
                    continue
            else:
                for b in tool_uses:
                    tool_calls_log.append(
                        {
                            "name": getattr(b, "name", ""),
                            "args": dict(getattr(b, "input", {}) or {}),
                            "is_error": True,
                            "duration_ms": 0,
                            "error": "blocked_tool_requires_web_fetch",
                        }
                    )
                assistant_blocks = [
                    b for b in _content_to_dicts(resp.raw.content) if b.get("type") != "tool_use"
                ]
                next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
                if assistant_blocks:
                    next_messages.append({"role": "assistant", "content": assistant_blocks})
                next_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": _web_fetch_required_message(
                                    min_seen_urls=terminal_min_seen_urls,
                                    seen_urls=seen_urls,
                                    fetchable_urls=fetchable_urls,
                                    session_fetched_source_keys=session_fetched_source_keys,
                                    hint=terminal_requirement_hint,
                                ),
                            }
                        ],
                    }
                )
                current_spec = AgentCallSpec(
                    route=current_spec.route,
                    system_blocks=current_spec.system_blocks,
                    user_blocks=current_spec.user_blocks,
                    tools=current_spec.tools,
                    tool_choice=current_spec.tool_choice,
                    max_output_tokens=current_spec.max_output_tokens,
                    stop_sequences=current_spec.stop_sequences,
                    extra_messages=next_messages,
                    reasoning_effort=current_spec.reasoning_effort,
                )
                fetch_guard_blocks += 1
                continue

        tool_uses = tool_uses[:parallel_cap]
        kept_ids = {getattr(tu, "id", None) for tu in tool_uses}

        # Dispatch in parallel
        results = await asyncio.gather(
            *(_dispatch(client, registry, tu, ctx, tool_timeout_s) for tu in tool_uses),
            return_exceptions=False,
        )

        # Update fetched-source tracking + log
        context_results: list[dict[str, Any]] = []
        for tu, r in zip(tool_uses, results, strict=True):
            if _supporting_tool_succeeded(tu.name, r):
                successful_tool_names.add(tu.name)
            else:
                successful_tool_names.discard(tu.name)
            context_r = _result_for_model_context(
                tool_name=tu.name,
                result=r,
                session_fetched_source_keys=session_fetched_source_keys,
            )
            context_results.append(context_r)
            tool_call_record = {
                "name": tu.name,
                "args": tu.input,
                "is_error": r["is_error"],
                "duration_ms": r.get("duration_ms", 0),
            }
            if (
                tu.name == "capability_validate_workflow"
                and not r["is_error"]
                and isinstance(r.get("content"), dict)
            ):
                tool_call_record["result"] = r["content"]
            tool_calls_log.append(tool_call_record)
            if tu.name == "web_fetch":
                web_fetch_attempts += 1
            elif tu.name == "rag_retrieve_context":
                rag_retrieve_attempts += 1
            if tu.name != "web_fetch" and not context_r["is_error"]:
                _extend_unique(fetchable_urls, _extract_urls(context_r.get("content")))
            for candidate in _recordable_citation_candidates(tu.name, r):
                url = str(candidate.get("url") or "").strip()
                if not url or url.casefold() in citation_candidate_urls:
                    continue
                citation_candidates.append(candidate)
                citation_candidate_urls.add(url.casefold())
            for u in _recordable_seen_urls(tu.name, tu.input, r):
                seen_urls.add(u)
                key = _source_key(u)
                if key:
                    session_fetched_source_keys.add(key)

        # A terminal record can arrive before one of the workflow's mandatory
        # supporting tools. Once those prerequisites have succeeded, do not
        # ask the model to regenerate an already valid record. Invalid records
        # use the same bounded fresh finalizer used at loop exhaustion.
        if last_blocked_terminal_input is not None and last_blocked_terminal_name:
            remaining_required_tools = [
                name
                for name in terminal_required_tool_names
                if name in available_tool_names and name not in successful_tool_names
            ]
            if not remaining_required_tools:
                validation_errors = _terminal_schema_validation_errors(
                    tools=current_spec.tools,
                    terminal_name=last_blocked_terminal_name,
                    candidate_input=last_blocked_terminal_input,
                )
                if not validation_errors:
                    reused_response = _terminal_response_from_candidate(
                        resp,
                        terminal_name=last_blocked_terminal_name,
                        candidate_input=last_blocked_terminal_input,
                        marker="reused",
                    )
                    tool_calls_log.append(
                        {
                            "name": last_blocked_terminal_name,
                            "args": last_blocked_terminal_input,
                            "is_error": False,
                            "duration_ms": 0,
                            "reused_blocked_terminal": True,
                        }
                    )
                    return ToolLoopResult(
                        response=reused_response,
                        iterations=iterations,
                        tool_calls=tool_calls_log,
                        seen_urls=seen_urls,
                        citation_candidates=citation_candidates,
                        last_blocked_terminal_input=last_blocked_terminal_input,
                    )

                finalized = await _try_terminal_finalization(
                    client,
                    source_spec=current_spec,
                    ctx=ctx,
                    terminal_name=last_blocked_terminal_name,
                    candidate_input=last_blocked_terminal_input,
                    validation_errors=validation_errors,
                )
                tool_calls_log.extend(finalized.attempts)
                if finalized.response is not None:
                    return ToolLoopResult(
                        response=finalized.response,
                        iterations=iterations,
                        tool_calls=tool_calls_log,
                        seen_urls=seen_urls,
                        citation_candidates=citation_candidates,
                        last_blocked_terminal_input=last_blocked_terminal_input,
                    )

        # Build next-turn spec: append the assistant message + a single user message
        # carrying all tool_result blocks. The assistant message must only carry
        # the tool_use blocks we actually dispatched — Anthropic requires every
        # tool_use to be paired with exactly one tool_result on the next turn.
        assistant_blocks = _content_to_dicts(resp.raw.content)
        assistant_blocks = [
            b for b in assistant_blocks if b.get("type") != "tool_use" or b.get("id") in kept_ids
        ]
        next_messages: list[dict[str, Any]] = list(current_spec.extra_messages)
        next_messages.append({"role": "assistant", "content": assistant_blocks})
        next_messages.append(
            {
                "role": "user",
                "content": [
                    _tool_result_block(tu, r)
                    for tu, r in zip(tool_uses, context_results, strict=True)
                ],
            }
        )
        current_spec = AgentCallSpec(
            route=current_spec.route,
            system_blocks=current_spec.system_blocks,
            user_blocks=current_spec.user_blocks,
            tools=current_spec.tools,
            tool_choice=current_spec.tool_choice,
            max_output_tokens=current_spec.max_output_tokens,
            stop_sequences=current_spec.stop_sequences,
            extra_messages=next_messages,
            reasoning_effort=current_spec.reasoning_effort,
        )

    remaining_required_tools = [
        name
        for name in terminal_required_tool_names
        if name in _tool_names(current_spec.tools) and name not in successful_tool_names
    ]
    if (
        force_terminal_tool
        and supports_terminal_finalization
        and not remaining_required_tools
        and not _needs_more_seen_urls(
            min_seen_urls=terminal_min_seen_urls,
            seen_urls=seen_urls,
            fetchable_urls=fetchable_urls,
            session_fetched_source_keys=session_fetched_source_keys,
            tools=current_spec.tools,
            web_fetch_attempts=web_fetch_attempts,
            guard_blocks=fetch_guard_blocks,
        )
    ):
        terminal_name = last_blocked_terminal_name or force_terminal_tool
        validation_errors = (
            _terminal_schema_validation_errors(
                tools=current_spec.tools,
                terminal_name=terminal_name,
                candidate_input=last_blocked_terminal_input,
            )
            if last_blocked_terminal_input is not None
            else []
        )
        finalized = await _try_terminal_finalization(
            client,
            source_spec=current_spec,
            ctx=ctx,
            terminal_name=terminal_name,
            candidate_input=last_blocked_terminal_input,
            validation_errors=validation_errors,
        )
        tool_calls_log.extend(finalized.attempts)
        if finalized.response is not None:
            return ToolLoopResult(
                response=finalized.response,
                iterations=iterations,
                tool_calls=tool_calls_log,
                seen_urls=seen_urls,
                citation_candidates=citation_candidates,
                last_blocked_terminal_input=last_blocked_terminal_input,
            )

    assert last is not None
    raise ToolLoopExhausted(ctx.agent, iterations)


# --------------------------------------------------------------------------- #
# helpers


def _terminal_schema_validation_errors(
    *,
    tools: list[dict[str, Any]],
    terminal_name: str,
    candidate_input: dict[str, Any] | None,
) -> list[str]:
    tool = next((item for item in tools if item.get("name") == terminal_name), None)
    if tool is None:
        return [f"terminal tool {terminal_name!r} is not in the active tool schema"]
    if candidate_input is None:
        return ["terminal candidate is missing"]
    schema = tool.get("input_schema") or {}
    try:
        errors = sorted(
            Draft202012Validator(schema).iter_errors(candidate_input),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
    except Exception as exc:
        return [f"terminal tool schema is invalid: {exc}"]
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '$'}: {error.message}"
        for error in errors
    ]


def _terminal_response_from_candidate(
    base: AnthropicResponse,
    *,
    terminal_name: str,
    candidate_input: dict[str, Any],
    marker: str,
) -> AnthropicResponse:
    raw = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=f"call_{marker}_{tool_run_id()}",
                name=terminal_name,
                input=candidate_input,
            )
        ],
        stop_reason="tool_use",
        usage=getattr(base.raw, "usage", None),
        model=getattr(base.raw, "model", ""),
        id=getattr(base.raw, "id", ""),
    )
    return AnthropicResponse(
        raw=raw,
        transcript_id=base.transcript_id,
        cost_usd=base.cost_usd,
        input_tokens=base.input_tokens,
        output_tokens=base.output_tokens,
        cache_read=base.cache_read,
        cache_write=base.cache_write,
    )


def _terminal_input_from_response(
    response: AnthropicResponse | None,
    *,
    terminal_name: str,
) -> dict[str, Any] | None:
    if response is None:
        return None
    for block in getattr(response.raw, "content", []) or []:
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == terminal_name
        ):
            value = getattr(block, "input", None)
            return dict(value) if isinstance(value, dict) else None
    return None


async def _try_terminal_finalization(
    client: AnthropicClient,
    *,
    source_spec: AgentCallSpec,
    ctx: CallContext,
    terminal_name: str,
    candidate_input: dict[str, Any] | None,
    validation_errors: list[str] | None = None,
    max_attempts: int = 2,
) -> _TerminalFinalizationResult:
    """Run a bounded fresh terminal synthesis/repair sequence."""
    if getattr(client, "supports_terminal_record_finalization", False) is not True:
        return _TerminalFinalizationResult(
            response=None,
            candidate_input=candidate_input,
            validation_errors=list(validation_errors or []),
            attempts=[],
        )

    current_candidate = candidate_input
    current_errors = list(validation_errors or [])
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, max_attempts) + 1):
        response = await client.finalize_terminal_record(
            source_spec=source_spec,
            ctx=ctx,
            terminal_tool_name=terminal_name,
            candidate_input=current_candidate,
            validation_errors=current_errors,
        )
        parsed = _terminal_input_from_response(response, terminal_name=terminal_name)
        errors = (
            _terminal_schema_validation_errors(
                tools=source_spec.tools,
                terminal_name=terminal_name,
                candidate_input=parsed,
            )
            if parsed is not None
            else ["finalization response did not contain the terminal tool"]
        )
        attempts.append(
            {
                "name": terminal_name,
                "args": parsed or current_candidate or {},
                "is_error": bool(errors),
                "duration_ms": 0,
                "terminal_finalization": True,
                "terminal_finalization_attempt": attempt,
                "terminal_finalization_mode": (
                    "repair" if current_candidate is not None else "synthesis"
                ),
                **({"validation_errors": errors} if errors else {}),
            }
        )
        if response is not None and parsed is not None and not errors:
            return _TerminalFinalizationResult(
                response=response,
                candidate_input=parsed,
                validation_errors=[],
                attempts=attempts,
            )
        if parsed is not None:
            current_candidate = parsed
        current_errors = errors

    return _TerminalFinalizationResult(
        response=None,
        candidate_input=current_candidate,
        validation_errors=current_errors,
        attempts=attempts,
    )


async def _dispatch(
    client: AnthropicClient, registry: ToolRegistry, tool_use, ctx: CallContext, timeout_s: float
) -> dict[str, Any]:
    """Run one tool call. Returns a dict with content + is_error + duration."""
    t0 = time.monotonic()
    run_id = tool_run_id()
    tctx = ToolCtx(
        cfg=registry._cfg,
        db=None,  # tools use their own write paths; DB writes go via repos
        session_id=ctx.session_id,
        task_id=ctx.task_id,
        run_id=run_id,
        llm_client=client,
        extra={"agent": ctx.agent, "action": ctx.action, "mode": ctx.mode},
    )
    args = dict(tool_use.input) if isinstance(tool_use.input, dict) else {"args": tool_use.input}
    try:
        result = await asyncio.wait_for(registry.call(tool_use.name, args, tctx), timeout=timeout_s)
    except TimeoutError:
        return {
            "is_error": True,
            "content": {"error": f"tool {tool_use.name!r} timed out"},
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
    return {
        "is_error": bool(result.is_error),
        "content": _tool_result_content(result),
        "duration_ms": result.duration_ms,
    }


def _tool_result_content(result) -> Any:
    if result.is_error:
        return {"error": result.error_message or "unknown error"}
    return result.content if result.content is not None else {"ok": True}


def _supporting_tool_succeeded(tool_name: str, result: dict[str, Any]) -> bool:
    if result.get("is_error"):
        return False
    content = result.get("content")
    if tool_name == "capability_get" and isinstance(content, dict):
        return bool(content.get("capabilities")) and not content.get("missing_ids")
    if tool_name == "capability_validate_workflow" and isinstance(content, dict):
        return str(content.get("status") or "") != "invalid"
    return True


def _result_for_model_context(
    *,
    tool_name: str,
    result: dict[str, Any],
    session_fetched_source_keys: set[str],
) -> dict[str, Any]:
    context_result = result
    if not result.get("is_error") and _is_search_tool(tool_name):
        context_result = {
            **result,
            "content": _compact_search_result_context(result.get("content")),
        }
    if (
        tool_name == "web_fetch"
        or context_result.get("is_error")
        or not session_fetched_source_keys
    ):
        return context_result
    content, omitted = _prune_session_duplicate_url_items(
        context_result.get("content"),
        session_fetched_source_keys,
    )
    if omitted <= 0:
        return context_result
    return {
        **context_result,
        "content": _annotate_session_duplicate_omissions(content, omitted),
    }


def _is_search_tool(tool_name: str) -> bool:
    return tool_name.endswith("_search") or tool_name == "web_search"


def _compact_search_result_context(content: Any) -> Any:
    if not isinstance(content, dict):
        return content
    results = content.get("results")
    if not isinstance(results, list):
        return content

    original_count = len(results)
    visible_results = results[:SEARCH_RESULTS_CONTEXT_LIMIT]
    compacted_results = [_first_author_only(item) for item in visible_results]
    omitted = max(0, original_count - len(compacted_results))
    authors_compacted = any(
        original is not compacted
        for original, compacted in zip(visible_results, compacted_results, strict=True)
    )
    if omitted <= 0 and not authors_compacted:
        return content

    out = dict(content)
    out["results"] = compacted_results
    out["n"] = len(compacted_results)
    out["total_results"] = content.get("n", original_count)
    out["model_context_results_limit"] = SEARCH_RESULTS_CONTEXT_LIMIT
    out["model_context_results_omitted"] = omitted
    out["model_context_note"] = (
        "Model context shows at most 30 search results and only the first author "
        "for each result. Full search records remain stored in artifacts and are "
        "used for PDF ingestion."
    )
    return out


def _first_author_only(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    authors = item.get("authors")
    if not isinstance(authors, list) or len(authors) <= 1:
        return item
    out = dict(item)
    out["authors"] = authors[:1]
    out["authors_omitted"] = len(authors) - 1
    return out


def _annotate_session_duplicate_omissions(content: Any, omitted: int) -> Any:
    note = (
        f"Omitted {omitted} result(s) whose fetch URL was already read earlier "
        "in this session. Prefer session-new URLs for source diversity."
    )
    if isinstance(content, dict):
        out = dict(content)
        out["session_duplicate_results_omitted"] = omitted
        out["source_diversity_note"] = note
        return out
    return {
        "results": content,
        "session_duplicate_results_omitted": omitted,
        "source_diversity_note": note,
    }


def _prune_session_duplicate_url_items(
    body: Any,
    session_fetched_source_keys: set[str],
) -> tuple[Any, int]:
    pruned, omitted, _drop = _prune_session_duplicate_url_node(
        body,
        session_fetched_source_keys,
        top_level=True,
    )
    return pruned, omitted


def _prune_session_duplicate_url_node(
    node: Any,
    session_fetched_source_keys: set[str],
    *,
    top_level: bool = False,
) -> tuple[Any, int, bool]:
    if isinstance(node, list):
        out = []
        omitted = 0
        for item in node:
            pruned, child_omitted, drop = _prune_session_duplicate_url_node(
                item,
                session_fetched_source_keys,
            )
            omitted += child_omitted
            if drop:
                omitted += 1
                continue
            out.append(pruned)
        return out, omitted, False
    if isinstance(node, dict):
        if not top_level and _dict_urls_all_session_duplicates(
            node,
            session_fetched_source_keys,
        ):
            return None, 0, True
        out: dict[str, Any] = {}
        omitted = 0
        for key, value in node.items():
            pruned, child_omitted, drop = _prune_session_duplicate_url_node(
                value,
                session_fetched_source_keys,
            )
            omitted += child_omitted
            if drop:
                omitted += 1
                continue
            out[key] = pruned
        return out, omitted, False
    return node, 0, False


def _dict_urls_all_session_duplicates(
    node: dict[str, Any],
    session_fetched_source_keys: set[str],
) -> bool:
    keys: list[str] = []
    for key_name in _URL_RE_KEYS:
        value = node.get(key_name)
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        source_key = _source_key(value)
        if not source_key:
            return False
        keys.append(source_key)
    return bool(keys) and all(key in session_fetched_source_keys for key in keys)


def _recordable_citation_candidates(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("is_error"):
        return []
    content = result.get("content")
    if tool_name == "rag_retrieve_context":
        return _citation_candidates_from_rag_content(content)
    if tool_name == "web_fetch" and isinstance(content, dict):
        url = str(content.get("url") or content.get("requested_url") or "").strip()
        if url.startswith(("http://", "https://")):
            return [{"url": url, "title": str(content.get("title") or url)}]
    return []


def _citation_candidates_from_rag_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, dict):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("source_documents", "sources", "rerank_chunks"):
        value = content.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            dedupe_key = url.casefold()
            if dedupe_key in seen:
                continue
            title = str(item.get("title") or item.get("source") or url).strip()
            out.append({"url": url, "title": title or url})
            seen.add(dedupe_key)
    return out


def _recordable_seen_urls(tool_name: str, tool_input: Any, result: dict[str, Any]) -> list[str]:
    if result.get("is_error"):
        return []
    content = result.get("content")
    if tool_name == "rag_retrieve_context":
        try:
            from ..tools.rag import recordable_rag_urls

            return recordable_rag_urls(content)
        except Exception:
            return []
    if tool_name != "web_fetch":
        return []
    if not isinstance(content, dict):
        return []
    text = content.get("text")
    fetched_url = content.get("url")
    if not (isinstance(text, str) and text.strip()):
        return []
    urls: list[str] = []
    if isinstance(fetched_url, str) and fetched_url.startswith(("http://", "https://")):
        urls.append(fetched_url)
    requested_content_url = content.get("requested_url")
    if isinstance(requested_content_url, str) and requested_content_url.startswith(
        ("http://", "https://")
    ):
        urls.append(requested_content_url)
    if isinstance(tool_input, dict):
        requested_url = tool_input.get("url")
        if isinstance(requested_url, str) and requested_url.startswith(("http://", "https://")):
            urls.append(requested_url)
    return list(dict.fromkeys(urls))


def _needs_more_seen_urls(
    *,
    min_seen_urls: int,
    seen_urls: set[str],
    fetchable_urls: list[str],
    session_fetched_source_keys: set[str],
    tools: list[dict[str, Any]],
    web_fetch_attempts: int,
    guard_blocks: int,
) -> bool:
    if min_seen_urls <= 0 or len(seen_urls) >= min_seen_urls:
        return False
    if "web_fetch" not in _tool_names(tools):
        return False
    max_fetch_enforcement = max(4, min_seen_urls * 3)
    if web_fetch_attempts >= max_fetch_enforcement or guard_blocks >= max_fetch_enforcement:
        return False
    return bool(_preferred_fetchable_urls(fetchable_urls, seen_urls, session_fetched_source_keys))


def _terminal_fetch_guard_message(
    *,
    terminal_name: str,
    min_seen_urls: int,
    seen_urls: set[str],
    fetchable_urls: list[str],
    session_fetched_source_keys: set[str],
    hint: str | None,
) -> str:
    needed = max(0, min_seen_urls - len(seen_urls))
    candidates = _preferred_fetchable_urls(
        fetchable_urls,
        seen_urls,
        session_fetched_source_keys,
    )[:8]
    candidate_block = "\n".join(f"- {url}" for url in candidates)
    msg = (
        f"You tried to call `{terminal_name}` before previewing enough sources. "
        f"Successful `web_fetch` previews so far: {len(seen_urls)}; required: {min_seen_urls}. "
        "Search results are discovery metadata only and are not citeable source text. "
        "When available, fetch URLs that have not already been previewed in this session. "
        f"Call `web_fetch` next on at least {needed} relevant distinct URL(s), then call "
        f"`{terminal_name}` after those fetches succeed."
    )
    if hint:
        msg += f"\n\n{hint}"
    if candidate_block:
        msg += f"\n\nFetchable URLs seen in search results include:\n{candidate_block}"
    return msg


def _terminal_required_tools_message(
    *,
    terminal_name: str,
    missing_tool_names: list[str],
    hint: str | None,
    candidate_input: dict[str, Any] | None = None,
) -> str:
    rendered = ", ".join(f"`{name}`" for name in missing_tool_names)
    msg = (
        f"You tried to call `{terminal_name}` before successfully completing required "
        f"supporting tool calls: {rendered}. Call the missing tools now, use their results in the "
        f"structured record, then call `{terminal_name}` again."
    )
    if hint:
        msg += f"\n\n{hint}"
    if candidate_input:
        candidate_json = json.dumps(candidate_input, ensure_ascii=True, sort_keys=True)
        msg += (
            "\n\nPreserve and repair this rejected candidate payload after completing "
            f"the supporting tools:\n{candidate_json[:16000]}"
        )
    return msg


def _web_fetch_required_message(
    *,
    min_seen_urls: int,
    seen_urls: set[str],
    fetchable_urls: list[str],
    session_fetched_source_keys: set[str],
    hint: str | None,
) -> str:
    needed = max(0, min_seen_urls - len(seen_urls))
    candidates = _preferred_fetchable_urls(
        fetchable_urls,
        seen_urls,
        session_fetched_source_keys,
    )[:8]
    candidate_block = "\n".join(f"- {url}" for url in candidates)
    msg = (
        "The previous search results include fetchable URLs, but there are not "
        "enough successful source previews yet. Search again later only if these "
        "candidates are irrelevant or fail to fetch. Prefer papers that have not "
        "already been previewed in this session. Call `web_fetch` next on at "
        f"least {needed} relevant distinct URL(s)."
    )
    if hint:
        msg += f"\n\n{hint}"
    if candidate_block:
        msg += f"\n\nFetchable URLs seen in search results include:\n{candidate_block}"
    return msg


def _session_new_fetch_required_message(
    *,
    min_seen_urls: int,
    seen_urls: set[str],
    fetchable_urls: list[str],
    session_fetched_source_keys: set[str],
    hint: str | None,
) -> str:
    needed = max(0, min_seen_urls - len(seen_urls))
    candidates = _preferred_fetchable_urls(
        fetchable_urls,
        seen_urls,
        session_fetched_source_keys,
    )[:8]
    msg = (
        "The requested `web_fetch` URL has already been fetched earlier in this session. "
        "To expand source diversity, use the search tools or fetch a relevant URL that has "
        "not yet been fetched in this session. "
    )
    if needed > 0:
        msg += f"Need {needed} successful new source preview(s) before recording."
    else:
        msg += (
            "If no relevant session-new source preview is needed, stop fetching and call the "
            "recording tool with citations supported by already-previewed sources."
        )
    if hint:
        msg += f"\n\n{hint}"
    if candidates:
        msg += "\n\nFetchable URLs not yet read in this session include:\n"
        msg += "\n".join(f"- {url}" for url in candidates)
    return msg


def _rag_first_blocked_discovery_uses(
    tool_uses: list[Any],
    *,
    cfg: Any,
    session_id: str | None,
    tools: list[dict[str, Any]],
    rag_retrieve_attempts: int,
    guard_blocks: int,
) -> list[Any]:
    if not _rag_retrieval_required(
        cfg=cfg,
        session_id=session_id,
        tools=tools,
        rag_retrieve_attempts=rag_retrieve_attempts,
        guard_blocks=guard_blocks,
    ):
        return []
    if any(getattr(tu, "name", "") == "rag_retrieve_context" for tu in tool_uses):
        return []
    return [tu for tu in tool_uses if getattr(tu, "name", "") in _DISCOVERY_TOOL_NAMES]


def _rag_retrieval_required(
    *,
    cfg: Any,
    session_id: str | None,
    tools: list[dict[str, Any]],
    rag_retrieve_attempts: int,
    guard_blocks: int,
) -> bool:
    if rag_retrieve_attempts > 0 or guard_blocks >= 3:
        return False
    names = _tool_names(tools)
    if "rag_retrieve_context" not in names:
        return False
    return _session_rag_index_ready(cfg, session_id)


def _session_rag_index_ready(cfg: Any, session_id: str | None) -> bool:
    if not session_id or not bool(getattr(getattr(cfg, "rag", None), "enabled", False)):
        return False
    session_rag_dir = getattr(cfg, "session_rag_dir", None)
    if not callable(session_rag_dir):
        return False
    try:
        root = session_rag_dir(session_id)
    except Exception:
        return False
    index_path = root / "kb.index"
    meta_path = root / "kb.pkl"
    manifest_path = root / "manifest.json"
    if not (index_path.is_file() and meta_path.is_file() and manifest_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    papers = manifest.get("papers")
    if not isinstance(papers, dict):
        return False
    return any(isinstance(p, dict) and p.get("indexed") for p in papers.values())


def _rag_retrieval_required_message(*, attempted_tool: str, terminal: bool) -> str:
    action = "recording a final structured result" if terminal else f"calling `{attempted_tool}`"
    return (
        f"The session RAG knowledge base already has indexed full-text papers, so before {action}, "
        "call `rag_retrieve_context` with a focused query for the hypothesis or mechanism under review. "
        "Use the returned chunks as citeable source text. After at least one RAG retrieval attempt, "
        "use broader search tools only if the retrieved context is missing needed evidence."
    )


_DISCOVERY_TOOL_NAMES = {
    "web_search",
    "arxiv_search",
    "biorxiv_search",
    "chemrxiv_search",
    "pubmed_search",
    "europe_pmc_search",
}


def _has_discovery_tool(tools: list[dict[str, Any]]) -> bool:
    return bool(_tool_names(tools) & _DISCOVERY_TOOL_NAMES)


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {t.get("name", "") for t in tools if isinstance(t.get("name"), str)}


def _extend_unique(target: list[str], urls: list[str]) -> None:
    seen = set(target)
    for url in urls:
        if url not in seen:
            target.append(url)
            seen.add(url)


def _preferred_fetchable_urls(
    fetchable_urls: list[str],
    seen_urls: set[str],
    session_fetched_source_keys: set[str],
) -> list[str]:
    session_new = _session_new_fetchable_urls(
        fetchable_urls,
        seen_urls,
        session_fetched_source_keys,
    )
    if session_new:
        return session_new
    return _unseen_fetchable_urls(fetchable_urls, seen_urls)


def _session_new_fetchable_urls(
    fetchable_urls: list[str],
    seen_urls: set[str],
    session_fetched_source_keys: set[str],
) -> list[str]:
    out: list[str] = []
    for url in _unseen_fetchable_urls(fetchable_urls, seen_urls):
        key = _source_key(url)
        if key and key not in session_fetched_source_keys:
            out.append(url)
    return out


def _unseen_fetchable_urls(fetchable_urls: list[str], seen_urls: set[str]) -> list[str]:
    return [url for url in fetchable_urls if url not in seen_urls]


def _session_duplicate_web_fetch_uses(
    tool_uses: list[Any],
    session_fetched_source_keys: set[str],
) -> list[Any]:
    out: list[Any] = []
    for tool_use in tool_uses:
        if getattr(tool_use, "name", "") != "web_fetch":
            continue
        key = _web_fetch_source_key(tool_use)
        if key and key in session_fetched_source_keys:
            out.append(tool_use)
    return out


def _web_fetch_source_key(tool_use: Any) -> str | None:
    tool_input = getattr(tool_use, "input", {}) or {}
    if not isinstance(tool_input, dict):
        return None
    url = tool_input.get("url")
    if not isinstance(url, str):
        return None
    return _source_key(url)


def _is_preferred_web_fetch(
    tool_use,
    *,
    seen_urls: set[str],
    session_fetched_source_keys: set[str],
    require_session_new: bool,
) -> bool:
    tool_input = getattr(tool_use, "input", {}) or {}
    if not isinstance(tool_input, dict):
        return not require_session_new
    url = tool_input.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return not require_session_new
    if url in seen_urls:
        return False
    if not require_session_new:
        return True
    key = _source_key(url)
    return bool(key and key not in session_fetched_source_keys)


def _session_fetched_source_keys(cfg: Any, session_id: str | None) -> set[str]:
    if session_id is None or not hasattr(cfg, "session_artifact_dir"):
        return set()
    papers_dir = cfg.session_artifact_dir(session_id) / "papers"
    if not papers_dir.exists():
        return set()
    keys: set[str] = set()
    for path in papers_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = payload.get("text")
        if not (isinstance(text, str) and text.strip()):
            continue
        for url_field in ("url", "requested_url"):
            url = payload.get(url_field)
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                key = _source_key(url)
                if key:
                    keys.add(key)
    return keys


_ARXIV_SOURCE_RE = re.compile(r"^/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?(?:\.pdf)?/?$")


def _source_key(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host == "arxiv.org":
        m = _ARXIV_SOURCE_RE.match(parts.path)
        if m:
            return f"arxiv:{m.group(1)}"
    path = parts.path.rstrip("/") or "/"
    return f"{parts.scheme.lower()}://{host}{path}"


def _tool_result_block(tool_use, r: dict[str, Any]) -> dict[str, Any]:
    body = r["content"]
    if getattr(tool_use, "name", "") == "web_fetch":
        body = _truncate_web_fetch_for_context(body)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use.id,
        "content": _content_to_text(body),
        "is_error": r["is_error"],
    }


def _truncate_web_fetch_for_context(body: Any) -> Any:
    if not isinstance(body, dict):
        return body
    text = body.get("text")
    if not isinstance(text, str) or len(text) <= WEB_FETCH_CONTEXT_TEXT_CHARS:
        return body
    out = {k: v for k, v in body.items() if k != "text"}
    out["truncated_for_context"] = True
    out["source_context_mode"] = "compact_excerpt"
    out["full_text_available_in_artifact_cache"] = True
    out["text"] = text[:WEB_FETCH_CONTEXT_TEXT_CHARS]
    return out


def _content_to_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, default=str, ensure_ascii=False)[:60_000]


def _content_to_dicts(content) -> list[dict[str, Any]]:
    """Convert SDK content blocks to plain dicts for re-sending.

    Thinking blocks must preserve their `signature` verbatim — Anthropic rejects
    a continuation turn that omits it.
    """
    out: list[dict[str, Any]] = []
    for b in content:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif t == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": getattr(b, "id", ""),
                    "name": getattr(b, "name", ""),
                    "input": getattr(b, "input", {}),
                }
            )
        elif t == "thinking":
            d: dict[str, Any] = {"type": "thinking", "thinking": getattr(b, "thinking", "")}
            sig = getattr(b, "signature", None)
            if sig:
                d["signature"] = sig
            out.append(d)
        elif t == "redacted_thinking":
            data = getattr(b, "data", None)
            if data:
                out.append({"type": "redacted_thinking", "data": data})
    return out


_URL_RE_KEYS = ("url", "abs_url", "pdf_url", "pubmed_url")


def _extract_urls(body: Any) -> list[str]:
    """Pull URLs out of nested tool_result content (best effort)."""
    out: list[str] = []
    _walk_urls(body, out)
    return out


def _walk_urls(node: Any, out: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in _URL_RE_KEYS and isinstance(v, str) and v.startswith(("http://", "https://")):
                out.append(v)
            else:
                _walk_urls(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_urls(item, out)
