# Modified from the original work.
"""Review repository."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from ...models import Review, ReviewScores


async def insert(conn: aiosqlite.Connection, r: Review) -> bool:
    cur = await conn.execute(
        """INSERT OR IGNORE INTO reviews(
               id, hypothesis_id, session_id, created_at, kind, verdict,
               novelty, correctness, testability, feasibility, body, artifact_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            r.id,
            r.hypothesis_id,
            r.session_id,
            r.created_at.isoformat(),
            r.kind,
            r.verdict,
            r.scores.novelty,
            r.scores.correctness,
            r.scores.testability,
            r.scores.feasibility,
            r.body,
            r.artifact_path,
        ),
    )
    inserted = cur.rowcount > 0
    await conn.commit()
    return inserted


async def upsert(conn: aiosqlite.Connection, r: Review) -> bool:
    """Insert or replace a deterministic review row.

    Reflection uses deterministic review IDs. A re-run after an older sparse
    review should repair that row instead of leaving the incomplete review in
    place. The plain insert() helper remains idempotent for tests/callers that
    need collision detection.
    """
    cur = await conn.execute(
        """INSERT INTO reviews(
               id, hypothesis_id, session_id, created_at, kind, verdict,
               novelty, correctness, testability, feasibility, body, artifact_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET
               hypothesis_id=excluded.hypothesis_id,
               session_id=excluded.session_id,
               created_at=excluded.created_at,
               kind=excluded.kind,
               verdict=excluded.verdict,
               novelty=excluded.novelty,
               correctness=excluded.correctness,
               testability=excluded.testability,
               feasibility=excluded.feasibility,
               body=excluded.body,
               artifact_path=excluded.artifact_path""",
        (
            r.id,
            r.hypothesis_id,
            r.session_id,
            r.created_at.isoformat(),
            r.kind,
            r.verdict,
            r.scores.novelty,
            r.scores.correctness,
            r.scores.testability,
            r.scores.feasibility,
            r.body,
            r.artifact_path,
        ),
    )
    await conn.commit()
    return cur.rowcount > 0


async def fetch(conn: aiosqlite.Connection, review_id: str) -> Review | None:
    async with conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)) as cur:
        row = await cur.fetchone()
    return _row_to_review(row) if row else None


async def list_for_hypothesis(conn: aiosqlite.Connection, hypothesis_id: str) -> list[Review]:
    async with conn.execute(
        "SELECT * FROM reviews WHERE hypothesis_id=? ORDER BY created_at DESC",
        (hypothesis_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_review(r) for r in rows]


async def list_for_session(conn: aiosqlite.Connection, session_id: str) -> list[Review]:
    async with conn.execute(
        "SELECT * FROM reviews WHERE session_id=? ORDER BY created_at DESC",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [_row_to_review(r) for r in rows]


def _row_to_review(row: aiosqlite.Row) -> Review:
    return Review(
        id=row["id"],
        hypothesis_id=row["hypothesis_id"],
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        kind=row["kind"],
        verdict=row["verdict"],
        scores=ReviewScores(
            novelty=row["novelty"],
            correctness=row["correctness"],
            testability=row["testability"],
            feasibility=row["feasibility"],
        ),
        assumptions=[],   # in JSON artifact
        evidence=[],      # in JSON artifact
        body=row["body"],
        artifact_path=row["artifact_path"],
    )
