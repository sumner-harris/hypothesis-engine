"""Session repository."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import aiosqlite

from ...models import ResearchPlan, Session


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def insert(conn: aiosqlite.Connection, s: Session) -> None:
    await conn.execute(
        """INSERT INTO sessions(
               id, created_at, updated_at, status, research_goal, research_plan,
               config_snapshot, budget_tokens, budget_usd, budget_used_tokens,
               budget_used_usd, wall_deadline, final_overview)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            s.id,
            s.created_at.isoformat(),
            s.updated_at.isoformat(),
            s.status,
            s.research_goal,
            s.research_plan.model_dump_json(),
            json.dumps(s.config_snapshot),
            s.budget_tokens,
            s.budget_usd,
            s.budget_used_tokens,
            s.budget_used_usd,
            s.wall_deadline.isoformat() if s.wall_deadline else None,
            s.final_overview,
        ),
    )
    await conn.commit()


async def fetch(conn: aiosqlite.Connection, session_id: str) -> Session | None:
    async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
        row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_session(row)


async def set_status(conn: aiosqlite.Connection, session_id: str, status: str) -> None:
    await conn.execute(
        "UPDATE sessions SET status=?, updated_at=? WHERE id=?",
        (status, _now(), session_id),
    )
    await conn.commit()


async def add_usage(
    conn: aiosqlite.Connection, session_id: str, tokens: int, usd: float
) -> None:
    await conn.execute(
        """UPDATE sessions
              SET budget_used_tokens = budget_used_tokens + ?,
                  budget_used_usd    = budget_used_usd    + ?,
                  updated_at         = ?
            WHERE id = ?""",
        (tokens, usd, _now(), session_id),
    )
    await conn.commit()


async def set_final_overview(conn: aiosqlite.Connection, session_id: str, rel_path: str) -> None:
    await conn.execute(
        "UPDATE sessions SET final_overview=?, status='done', updated_at=? WHERE id=?",
        (rel_path, _now(), session_id),
    )
    await conn.commit()


def _row_to_session(row: aiosqlite.Row) -> Session:
    return Session(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        status=row["status"],
        research_goal=row["research_goal"],
        research_plan=ResearchPlan.model_validate_json(row["research_plan"]),
        config_snapshot=json.loads(row["config_snapshot"]),
        budget_tokens=row["budget_tokens"],
        budget_usd=row["budget_usd"],
        budget_used_tokens=row["budget_used_tokens"],
        budget_used_usd=row["budget_used_usd"],
        wall_deadline=datetime.fromisoformat(row["wall_deadline"]) if row["wall_deadline"] else None,
        final_overview=row["final_overview"],
    )
