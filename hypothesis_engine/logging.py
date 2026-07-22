"""Structlog setup.

Emits JSONL to stdout and (when a session is bound) also to
`data/logs/session-<id>.jsonl`. The bound contextvars (session_id, task_id,
agent, trace_id, span_id) propagate to every log line for free.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

__all__ = [
    "add_session_sink",
    "bind",
    "clear",
    "get_logger",
    "setup_logging",
    "unbind",
]


def _add_ts(_logger, _name, event_dict):  # type: ignore[no-untyped-def]
    event_dict.setdefault("ts", int(time.time() * 1000))
    return event_dict


_processors_pre = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    _add_ts,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def setup_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            *_processors_pre,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()


# Per-session sink ----------------------------------------------------------- #

_session_sinks: dict[str, Any] = {}


def add_session_sink(session_id: str, log_path: Path) -> None:
    """Tee subsequent log lines to a per-session JSONL file via stdlib logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _session_sinks[session_id] = handler


def remove_session_sink(session_id: str) -> None:
    handler = _session_sinks.pop(session_id, None)
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        handler.close()


# Context helpers ----------------------------------------------------------- #


def bind(**kwargs: Any) -> None:
    bind_contextvars(**kwargs)


def unbind(*keys: str) -> None:
    unbind_contextvars(*keys)


def clear() -> None:
    clear_contextvars()
