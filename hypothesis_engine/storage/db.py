# Modified from the original work.
"""SQLite connection and schema-initialization helpers.

Single connection per process is fine for our workload (writes serialize on the WAL
anyway). We use aiosqlite for the async API and set PRAGMAs on each connect.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from ..config import Config

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def _apply_pragmas(conn: aiosqlite.Connection) -> None:
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    await conn.execute("PRAGMA foreign_keys=ON;")
    await conn.execute("PRAGMA busy_timeout=5000;")
    await conn.commit()


async def connect(cfg: Config) -> aiosqlite.Connection:
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(cfg.db_path)
    conn.row_factory = aiosqlite.Row
    await _apply_pragmas(conn)
    return conn


async def init_db(cfg: Config) -> None:
    """Create all current tables and indexes from the canonical schema."""
    conn = await connect(cfg)
    try:
        await conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        await conn.commit()
    finally:
        await conn.close()


async def table_names(conn: aiosqlite.Connection) -> list[str]:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cur:
        rows = await cur.fetchall()
    return [r["name"] for r in rows]
