"""Embeddings metadata repository.

Vectors themselves live in FAISS on disk. This table only stores pointers:
faiss_offset (index position), model, dim, text_hash (cache key).
"""

from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def upsert(
    conn: aiosqlite.Connection,
    *,
    id_: str,
    session_id: str,
    hypothesis_id: str,
    model: str,
    dim: int,
    faiss_offset: int,
    text_hash: str,
) -> None:
    await conn.execute(
        """INSERT INTO embeddings_meta(
               id, session_id, hypothesis_id, model, dim, faiss_offset, text_hash, created_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(hypothesis_id, model) DO UPDATE
              SET faiss_offset=excluded.faiss_offset,
                  text_hash=excluded.text_hash,
                  created_at=excluded.created_at""",
        (id_, session_id, hypothesis_id, model, dim, faiss_offset, text_hash, _now()),
    )
    await conn.commit()


async def has_embedding(
    conn: aiosqlite.Connection, hypothesis_id: str, model: str
) -> bool:
    async with conn.execute(
        "SELECT 1 FROM embeddings_meta WHERE hypothesis_id=? AND model=?",
        (hypothesis_id, model),
    ) as cur:
        return await cur.fetchone() is not None


async def list_for_session(
    conn: aiosqlite.Connection, session_id: str
) -> list[dict]:
    async with conn.execute(
        """SELECT id, hypothesis_id, model, dim, faiss_offset, text_hash
             FROM embeddings_meta
            WHERE session_id=?
            ORDER BY faiss_offset""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def fetch_offset(
    conn: aiosqlite.Connection, hypothesis_id: str, model: str
) -> int | None:
    async with conn.execute(
        "SELECT faiss_offset FROM embeddings_meta WHERE hypothesis_id=? AND model=?",
        (hypothesis_id, model),
    ) as cur:
        row = await cur.fetchone()
    return row["faiss_offset"] if row else None
