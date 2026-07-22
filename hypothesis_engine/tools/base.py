# Modified from the original work.
"""Tool protocol shared by all tools (built-in + science-skills + LLM hosted)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import aiosqlite

from ..config import Config


@dataclass
class ToolCtx:
    """Per-call context passed to every tool invocation.

    Tools should persist any non-trivial artifact (raw stdout, fetched paper)
    under the session's artifacts directory using this context.
    """

    cfg: Config
    db: aiosqlite.Connection | None = None
    session_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None       # ULID for this individual tool invocation
    llm_client: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Standard result envelope.

    `content` is what gets shown back to the LLM via tool_result. It should be
    short and structured (JSON-serializable Python). `artifact_path` is the
    relative on-disk path where the raw, large output was persisted.
    """

    is_error: bool = False
    content: Any = None
    artifact_path: str | None = None
    error_message: str | None = None
    duration_ms: int = 0
    result_bytes: int = 0


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]   # JSONSchema dict (Anthropic tool format)

    async def call(self, args: dict[str, Any], ctx: ToolCtx) -> ToolResult: ...


def to_anthropic_tool(t: Tool) -> dict[str, Any]:
    """Render a Tool as the dict Anthropic's `tools` param expects."""
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }
