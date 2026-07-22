# Modified from the original work.
"""System / human feedback repository."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from ...models import SystemFeedback


async def insert(conn: aiosqlite.Connection, f: SystemFeedback) -> None:
    await conn.execute(
        """INSERT INTO system_feedback(
               id, session_id, created_at, source, kind, target_id, text,
               artifact_path, active)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            f.id, f.session_id, f.created_at.isoformat(),
            f.source, f.kind, f.target_id, f.text,
            f.artifact_path, 1 if f.active else 0,
        ),
    )
    await conn.commit()


async def active_for_session(
    conn: aiosqlite.Connection, session_id: str, target_id: str | None = None
) -> list[SystemFeedback]:
    """Return active feedback. If `target_id` is given, include targeted matches *plus* global."""
    if target_id:
        async with conn.execute(
            """SELECT * FROM system_feedback
                  WHERE session_id=? AND active=1
                    AND (target_id IS NULL OR target_id=?)
                  ORDER BY created_at DESC""",
            (session_id, target_id),
        ) as cur:
            rows = await cur.fetchall()
    else:
        async with conn.execute(
            """SELECT * FROM system_feedback
                  WHERE session_id=? AND active=1 AND target_id IS NULL
                  ORDER BY created_at DESC""",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_fb(r) for r in rows]


async def active_human_for_session(
    conn: aiosqlite.Connection, session_id: str
) -> list[SystemFeedback]:
    async with conn.execute(
        """SELECT * FROM system_feedback
              WHERE session_id=? AND active=1 AND source='human'
              ORDER BY created_at DESC""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_fb(r) for r in rows]


async def human_preferences_for_session(
    conn: aiosqlite.Connection, session_id: str
) -> list[SystemFeedback]:
    """Return every global human preference in submission order for prompt inspection."""
    async with conn.execute(
        """SELECT * FROM system_feedback
              WHERE session_id=? AND source='human' AND kind='preference'
                AND target_id IS NULL
              ORDER BY created_at ASC""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_fb(r) for r in rows]


async def latest_system_feedback(
    conn: aiosqlite.Connection, session_id: str
) -> SystemFeedback | None:
    async with conn.execute(
        """SELECT * FROM system_feedback
              WHERE session_id=? AND kind='system_feedback' AND source='meta_review'
              ORDER BY created_at DESC LIMIT 1""",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    return _row_to_fb(row) if row else None


async def deactivate(conn: aiosqlite.Connection, feedback_id: str) -> None:
    await conn.execute(
        "UPDATE system_feedback SET active=0 WHERE id=?", (feedback_id,)
    )
    await conn.commit()


def _row_to_fb(row: aiosqlite.Row) -> SystemFeedback:
    return SystemFeedback(
        id=row["id"],
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        source=row["source"],
        kind=row["kind"],
        target_id=row["target_id"],
        text=row["text"],
        artifact_path=row["artifact_path"],
        active=bool(row["active"]),
    )
