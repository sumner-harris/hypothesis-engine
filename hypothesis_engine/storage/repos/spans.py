"""Span repository — OpenTelemetry-shaped spans persisted into SQLite.

We keep the schema OTel-compatible so a future exporter can stream rows out
without re-shaping. For now everything is one-process and lives here.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from ...ids import span_id as new_span_id


async def start(
    conn: aiosqlite.Connection,
    *,
    trace_id: str,
    parent_span_id: str | None,
    session_id: str | None,
    task_id: str | None,
    name: str,
    attrs: dict[str, Any] | None = None,
) -> str:
    sid = new_span_id()
    await conn.execute(
        """INSERT INTO spans(span_id, trace_id, parent_span_id, session_id, task_id,
                              name, started_at, ended_at, attrs_json, status)
           VALUES (?,?,?,?,?,?,?,NULL,?,'unset')""",
        (
            sid, trace_id, parent_span_id, session_id, task_id, name,
            int(time.time() * 1000), json.dumps(attrs or {}),
        ),
    )
    await conn.commit()
    return sid


async def end(
    conn: aiosqlite.Connection,
    span_id_: str,
    *,
    status: str = "ok",
    attrs: dict[str, Any] | None = None,
) -> None:
    if attrs is not None:
        # merge attrs
        async with conn.execute("SELECT attrs_json FROM spans WHERE span_id=?", (span_id_,)) as cur:
            row = await cur.fetchone()
        prev = json.loads(row["attrs_json"]) if row and row["attrs_json"] else {}
        prev.update(attrs)
        attrs_json: str | None = json.dumps(prev)
    else:
        attrs_json = None
    if attrs_json is None:
        await conn.execute(
            "UPDATE spans SET ended_at=?, status=? WHERE span_id=?",
            (int(time.time() * 1000), status, span_id_),
        )
    else:
        await conn.execute(
            "UPDATE spans SET ended_at=?, status=?, attrs_json=? WHERE span_id=?",
            (int(time.time() * 1000), status, attrs_json, span_id_),
        )
    await conn.commit()


@asynccontextmanager
async def span(
    conn: aiosqlite.Connection,
    *,
    trace_id: str,
    parent_span_id: str | None,
    session_id: str | None,
    task_id: str | None,
    name: str,
    attrs: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    sid = await start(
        conn,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        session_id=session_id,
        task_id=task_id,
        name=name,
        attrs=attrs,
    )
    try:
        yield sid
        await end(conn, sid, status="ok")
    except Exception as exc:
        await end(conn, sid, status="error", attrs={"error": str(exc)})
        raise
