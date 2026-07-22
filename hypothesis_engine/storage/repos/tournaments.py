# Modified from the original work.
"""Tournament matches + Elo journal.

The Elo update is the *only* place hypotheses.elo and matches_played mutate
during a session. It runs in a single transaction guarded by elo_journal.match_id
UNIQUE — so re-running the same match is a no-op.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime

import aiosqlite

from ...models import TournamentMatch
from ...orchestrator.elo import update_elo


async def insert_match(conn: aiosqlite.Connection, m: TournamentMatch) -> bool:
    """Insert the descriptive match row. Idempotent by id."""
    cur = await conn.execute(
        """INSERT OR IGNORE INTO tournament_matches(
               id, session_id, created_at, hyp_a, hyp_b, mode, winner,
               elo_a_before, elo_b_before, elo_a_after, elo_b_after,
               rationale, transcript_id, similarity,
               prompt1_hyp_id, prompt2_hyp_id, prompt1_side, prompt2_side,
               winner_prompt_position, prompt1_chars, prompt2_chars, prompt_order_key)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            m.id, m.session_id, m.created_at.isoformat(),
            m.hyp_a, m.hyp_b, m.mode, m.winner,
            m.elo_a_before, m.elo_b_before, m.elo_a_after, m.elo_b_after,
            m.rationale, m.transcript_id, m.similarity,
            m.prompt1_hyp_id, m.prompt2_hyp_id, m.prompt1_side, m.prompt2_side,
            m.winner_prompt_position, m.prompt1_chars, m.prompt2_chars, m.prompt_order_key,
        ),
    )
    ok = cur.rowcount > 0
    await conn.commit()
    return ok


async def apply_elo_update(
    conn: aiosqlite.Connection,
    *,
    match_id: str,
    hyp_a: str,
    hyp_b: str,
    winner: str,
    elo_a_before: float,
    elo_b_before: float,
    elo_a_after: float | None = None,
    elo_b_after: float | None = None,
    k_new: int = 32,
    k_warm: int = 16,
    logistic_scale: float = 400.0,
    elo_initial: float = 1200.0,
) -> bool:
    """Apply an Elo update atomically and idempotently.

    The ranking worker may have selected a pair and called the LLM using stale
    Elo values while other workers were completing matches. Recompute the final
    Elo update from the current DB values inside this write transaction so
    parallel ranking cannot overwrite intervening updates with stale ratings.

    Returns True if the update was newly applied; False if the journal already
    has this match_id (re-run; we skip).
    """
    _ = (elo_a_before, elo_b_before, elo_a_after, elo_b_after)
    applied_at = int(time.time() * 1000)
    tx = await _transaction_connection(conn)
    try:
        await tx.execute("BEGIN IMMEDIATE")

        current: dict[str, aiosqlite.Row] = {}
        async with tx.execute(
            "SELECT id, elo, matches_played FROM hypotheses WHERE id IN (?,?)",
            (hyp_a, hyp_b),
        ) as cur:
            rows = await cur.fetchall()
        current = {row["id"]: row for row in rows}
        if hyp_a not in current or hyp_b not in current:
            missing = [hid for hid in (hyp_a, hyp_b) if hid not in current]
            raise RuntimeError(f"cannot apply Elo update; missing hypotheses: {missing}")

        current_a = current[hyp_a]
        current_b = current[hyp_b]
        applied_a_before = float(current_a["elo"] if current_a["elo"] is not None else elo_initial)
        applied_b_before = float(current_b["elo"] if current_b["elo"] is not None else elo_initial)
        matches_played_min = min(
            int(current_a["matches_played"] or 0),
            int(current_b["matches_played"] or 0),
        )
        update = update_elo(
            applied_a_before,
            applied_b_before,
            winner,
            matches_played_min=matches_played_min,
            k_new=k_new,
            k_warm=k_warm,
            logistic_scale=logistic_scale,
        )

        await tx.execute(
            """INSERT INTO elo_journal(
                   update_id, match_id, hyp_a, hyp_b, winner,
                   elo_a_before, elo_b_before, elo_a_after, elo_b_after, applied_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                match_id,
                match_id,
                hyp_a,
                hyp_b,
                winner,
                update.elo_a_before,
                update.elo_b_before,
                update.elo_a_after,
                update.elo_b_after,
                applied_at,
            ),
        )
        await tx.execute(
            """UPDATE hypotheses
                  SET elo=?, matches_played=matches_played+1
                WHERE id=?""",
            (update.elo_a_after, hyp_a),
        )
        await tx.execute(
            """UPDATE hypotheses
                  SET elo=?, matches_played=matches_played+1
                WHERE id=?""",
            (update.elo_b_after, hyp_b),
        )
        await tx.execute(
            """UPDATE tournament_matches
                  SET elo_a_before=?, elo_b_before=?, elo_a_after=?, elo_b_after=?
                WHERE id=?""",
            (
                update.elo_a_before,
                update.elo_b_before,
                update.elo_a_after,
                update.elo_b_after,
                match_id,
            ),
        )
        await tx.commit()
        return True
    except aiosqlite.IntegrityError:
        # match_id already in journal — idempotent skip.
        await tx.rollback()
        return False
    except Exception:
        await tx.rollback()
        raise
    finally:
        await tx.close()


async def active_pair_keys(conn: aiosqlite.Connection, session_id: str) -> set[str]:
    now_ms = int(time.time() * 1000)
    await _cleanup_inactive_pair_reservations(conn, session_id, now_ms)
    async with conn.execute(
        """SELECT pair_key
              FROM tournament_pair_reservations
             WHERE session_id=?
             ORDER BY pair_key""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()
    return {row["pair_key"] for row in rows}


async def reserve_pair(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    pair_key: str,
    task_id: str,
    max_pair_matches: int = 3,
    wins_to_close_pair: int = 2,
) -> bool:
    now_ms = int(time.time() * 1000)
    tx = await _transaction_connection(conn)
    try:
        await tx.execute("BEGIN IMMEDIATE")
        await _cleanup_inactive_pair_reservations(tx, session_id, now_ms, commit=False)
        async with tx.execute(
            "SELECT status, lease_expires_at FROM tasks WHERE id=? AND session_id=?",
            (task_id, session_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row["status"] != "in_progress":
            await tx.rollback()
            return False
        lease_expires_at = row["lease_expires_at"]
        if lease_expires_at is not None and int(lease_expires_at) <= now_ms:
            await tx.rollback()
            return False
        if await _pair_closed(
            tx,
            session_id,
            pair_key,
            max_pair_matches=max_pair_matches,
            wins_to_close_pair=wins_to_close_pair,
        ):
            await tx.rollback()
            return False
        cur = await tx.execute(
            """INSERT OR IGNORE INTO tournament_pair_reservations(
                   session_id, pair_key, task_id, reserved_at)
               VALUES (?,?,?,?)""",
            (session_id, pair_key, task_id, now_ms),
        )
        ok = cur.rowcount > 0
        await tx.commit()
        return ok
    except Exception:
        await tx.rollback()
        raise
    finally:
        await tx.close()


async def release_pair(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    pair_key: str,
    task_id: str,
) -> None:
    await conn.execute(
        """DELETE FROM tournament_pair_reservations
             WHERE session_id=? AND pair_key=? AND task_id=?""",
        (session_id, pair_key, task_id),
    )
    await conn.commit()


async def _cleanup_inactive_pair_reservations(
    conn: aiosqlite.Connection,
    session_id: str,
    now_ms: int,
    *,
    commit: bool = True,
) -> None:
    await conn.execute(
        """DELETE FROM tournament_pair_reservations
             WHERE session_id=?
               AND NOT EXISTS (
                   SELECT 1
                     FROM tasks t
                    WHERE t.id=tournament_pair_reservations.task_id
                      AND t.session_id=tournament_pair_reservations.session_id
                      AND t.status='in_progress'
                      AND (t.lease_expires_at IS NULL OR t.lease_expires_at > ?)
               )""",
        (session_id, now_ms),
    )
    if commit:
        await conn.commit()


async def closed_pair_keys(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    max_pair_matches: int = 3,
    wins_to_close_pair: int = 2,
) -> set[str]:
    """Return unordered pair keys whose best-of-N contest is already resolved."""
    async with conn.execute(
        """SELECT hyp_a, hyp_b, winner
             FROM tournament_matches
            WHERE session_id=?
              AND mode != 'invalid'
              AND winner IS NOT NULL""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()

    total_by_pair: dict[str, int] = {}
    wins_by_pair: dict[str, dict[str, int]] = {}
    for row in rows:
        pair_key = _pair_key(row["hyp_a"], row["hyp_b"])
        total_by_pair[pair_key] = total_by_pair.get(pair_key, 0) + 1
        winner_id = _winner_hypothesis_id(row)
        if winner_id is None:
            continue
        wins = wins_by_pair.setdefault(pair_key, {})
        wins[winner_id] = wins.get(winner_id, 0) + 1

    closed: set[str] = set()
    for pair_key, total in total_by_pair.items():
        if total >= max_pair_matches:
            closed.add(pair_key)
            continue
        if any(wins >= wins_to_close_pair for wins in wins_by_pair.get(pair_key, {}).values()):
            closed.add(pair_key)
    return closed


async def eligible_pair_count(
    conn: aiosqlite.Connection,
    session_id: str,
    hypothesis_ids: list[str],
    *,
    max_pair_matches: int = 3,
    wins_to_close_pair: int = 2,
    exclude_active: bool = True,
) -> int:
    if len(hypothesis_ids) < 2:
        return 0
    closed = await closed_pair_keys(
        conn,
        session_id,
        max_pair_matches=max_pair_matches,
        wins_to_close_pair=wins_to_close_pair,
    )
    active = await active_pair_keys(conn, session_id) if exclude_active else set()
    ids = sorted(set(hypothesis_ids))
    n = 0
    for idx, hyp_a in enumerate(ids):
        for hyp_b in ids[idx + 1:]:
            pair_key = _pair_key(hyp_a, hyp_b)
            if pair_key in closed or pair_key in active:
                continue
            n += 1
    return n


async def next_pair_orientation(
    conn: aiosqlite.Connection,
    session_id: str,
    hyp_a: str,
    hyp_b: str,
) -> tuple[str, str]:
    """Return the storage/prompt A-B order for the next valid pair round.

    Repeated pair rounds alternate relative to the previous valid match. New
    pairs use a deterministic hash so the initial side is balanced but stable
    across retries.
    """
    rows = await _valid_pair_rows(conn, session_id, hyp_a, hyp_b)
    if rows:
        last = rows[-1]
        return last["hyp_b"], last["hyp_a"]

    first, second = sorted((hyp_a, hyp_b))
    key = _pair_key(first, second)
    digest = hashlib.sha256(
        f"{session_id}:{key}:initial-pair-orientation".encode()
    ).hexdigest()
    if int(digest[:8], 16) % 2 == 0:
        return first, second
    return second, first


async def pair_status(
    conn: aiosqlite.Connection,
    session_id: str,
    hyp_a: str,
    hyp_b: str,
    *,
    max_pair_matches: int = 3,
    wins_to_close_pair: int = 2,
) -> dict[str, object]:
    rows = await _valid_pair_rows(conn, session_id, hyp_a, hyp_b)
    wins: dict[str, int] = {}
    for row in rows:
        winner_id = _winner_hypothesis_id(row)
        if winner_id is None:
            continue
        wins[winner_id] = wins.get(winner_id, 0) + 1

    winner_hyp_id = None
    for hid, n_wins in wins.items():
        if n_wins >= wins_to_close_pair:
            winner_hyp_id = hid
            break
    if winner_hyp_id is None and len(rows) >= max_pair_matches and wins:
        winner_hyp_id = max(wins.items(), key=lambda item: item[1])[0]

    closed = winner_hyp_id is not None or len(rows) >= max_pair_matches
    return {
        "closed": closed,
        "winner_hyp_id": winner_hyp_id,
        "valid_matches": len(rows),
        "wins": wins,
        "max_pair_matches": max_pair_matches,
        "wins_to_close_pair": wins_to_close_pair,
    }


async def _pair_closed(
    conn: aiosqlite.Connection,
    session_id: str,
    pair_key: str,
    *,
    max_pair_matches: int,
    wins_to_close_pair: int,
) -> bool:
    try:
        hyp_a, hyp_b = pair_key.split("::", 1)
    except ValueError:
        return False
    status = await pair_status(
        conn,
        session_id,
        hyp_a,
        hyp_b,
        max_pair_matches=max_pair_matches,
        wins_to_close_pair=wins_to_close_pair,
    )
    return bool(status["closed"])


async def _valid_pair_rows(
    conn: aiosqlite.Connection,
    session_id: str,
    hyp_a: str,
    hyp_b: str,
) -> list[aiosqlite.Row]:
    async with conn.execute(
        """SELECT id, created_at, hyp_a, hyp_b, winner
             FROM tournament_matches
            WHERE session_id=?
              AND mode != 'invalid'
              AND winner IS NOT NULL
              AND ((hyp_a=? AND hyp_b=?) OR (hyp_a=? AND hyp_b=?))
            ORDER BY created_at ASC, id ASC""",
        (session_id, hyp_a, hyp_b, hyp_b, hyp_a),
    ) as cur:
        return await cur.fetchall()


def _pair_key(hyp_a: str, hyp_b: str) -> str:
    first, second = sorted((hyp_a, hyp_b))
    return f"{first}::{second}"


def _winner_hypothesis_id(row: aiosqlite.Row) -> str | None:
    if row["winner"] == "a":
        return row["hyp_a"]
    if row["winner"] == "b":
        return row["hyp_b"]
    return None


async def _transaction_connection(conn: aiosqlite.Connection) -> aiosqlite.Connection:
    async with conn.execute("PRAGMA database_list") as cur:
        rows = await cur.fetchall()
    db_path = next((row["file"] for row in rows if row["name"] == "main"), None)
    if not db_path:
        raise RuntimeError("cannot open isolated Elo transaction for unnamed database")

    tx = await aiosqlite.connect(db_path)
    tx.row_factory = aiosqlite.Row
    await tx.execute("PRAGMA journal_mode=WAL;")
    await tx.execute("PRAGMA synchronous=NORMAL;")
    await tx.execute("PRAGMA foreign_keys=ON;")
    await tx.execute("PRAGMA busy_timeout=5000;")
    await tx.commit()
    return tx


async def count_matches(conn: aiosqlite.Connection, session_id: str) -> int:
    async with conn.execute(
        "SELECT COUNT(*) AS n FROM tournament_matches WHERE session_id=? AND mode != 'invalid'",
        (session_id,),
    ) as cur:
        row = await cur.fetchone()
    return row["n"] if row else 0


async def recent_rationales(
    conn: aiosqlite.Connection, session_id: str, limit: int = 50
) -> list[str]:
    async with conn.execute(
        """SELECT rationale FROM tournament_matches
              WHERE session_id=? AND rationale IS NOT NULL
              ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [r["rationale"] for r in rows]


def _row_to_match(row: aiosqlite.Row) -> TournamentMatch:
    return TournamentMatch(
        id=row["id"],
        session_id=row["session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        hyp_a=row["hyp_a"], hyp_b=row["hyp_b"], mode=row["mode"],
        winner=row["winner"],
        elo_a_before=row["elo_a_before"], elo_b_before=row["elo_b_before"],
        elo_a_after=row["elo_a_after"], elo_b_after=row["elo_b_after"],
        rationale=row["rationale"], transcript_id=row["transcript_id"],
        similarity=row["similarity"],
        prompt1_hyp_id=_row_get(row, "prompt1_hyp_id"),
        prompt2_hyp_id=_row_get(row, "prompt2_hyp_id"),
        prompt1_side=_row_get(row, "prompt1_side"),
        prompt2_side=_row_get(row, "prompt2_side"),
        winner_prompt_position=_row_get(row, "winner_prompt_position"),
        prompt1_chars=_row_get(row, "prompt1_chars"),
        prompt2_chars=_row_get(row, "prompt2_chars"),
        prompt_order_key=_row_get(row, "prompt_order_key"),
    )


def _row_get(row: aiosqlite.Row, key: str):
    try:
        return row[key]
    except (IndexError, KeyError):
        return None
