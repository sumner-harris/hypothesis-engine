"""Actions triggered by human feedback."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .. import ids
from ..models import Task
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import tasks as task_repo


def _complete_full_review(review) -> bool:
    return (
        review.kind == "full"
        and review.verdict is not None
        and review.scores.novelty is not None
        and review.scores.correctness is not None
        and review.scores.testability is not None
        and review.scores.feasibility is not None
    )


async def _enqueue_metareview(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    feedback_id: str,
    kind: str,
    target_id: str | None,
) -> bool:
    return await task_repo.enqueue(conn, Task(
        id=ids.task_id(),
        session_id=session_id,
        created_at=datetime.now(UTC),
        agent="metareview",
        action="GenerateSystemFeedback",
        target_id=None,
        payload={
            "reason": "human_feedback",
            "feedback_id": feedback_id,
            "feedback_kind": kind,
            "target_id": target_id,
        },
        priority=30,
        status="pending",
        idempotency_key=f"{session_id}::metareview::human_feedback::{feedback_id}",
    ))


async def apply_human_feedback_actions(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    feedback_id: str,
    kind: str,
    target_id: str | None,
) -> dict[str, Any]:
    """Apply direct side effects of human feedback and enqueue follow-up work."""
    enqueued = 0
    tasks: list[str] = []

    if await _enqueue_metareview(
        conn,
        session_id=session_id,
        feedback_id=feedback_id,
        kind=kind,
        target_id=target_id,
    ):
        enqueued += 1
        tasks.append("metareview")

    if not target_id:
        return {"enqueued": enqueued, "tasks": tasks}

    if kind == "rejection":
        h = await hyp_repo.fetch(conn, target_id)
        if h is None or h.session_id != session_id:
            return {
                "enqueued": enqueued,
                "tasks": tasks,
                "missing_target": True,
            }
        await hyp_repo.set_state(conn, target_id, "rejected")
        cancelled = await task_repo.cancel_pending_for_target(conn, session_id, target_id)
        return {
            "enqueued": enqueued,
            "tasks": tasks,
            "state": "rejected",
            "cancelled_pending": cancelled,
        }

    if kind != "pin":
        return {"enqueued": enqueued, "tasks": tasks}

    h = await hyp_repo.fetch(conn, target_id)
    if h is None or h.session_id != session_id:
        return {
            "enqueued": enqueued,
            "tasks": tasks,
            "missing_target": True,
        }

    await hyp_repo.set_state(conn, target_id, "pinned")
    reviews = await rev_repo.list_for_hypothesis(conn, target_id)
    has_complete_review = any(_complete_full_review(r) for r in reviews)

    if not has_complete_review:
        inserted = await task_repo.enqueue(conn, Task(
            id=ids.task_id(),
            session_id=session_id,
            created_at=datetime.now(UTC),
            agent="reflection",
            action="ReviewHypothesis",
            target_id=target_id,
            payload={"kind": "full", "reason": "human_pin", "feedback_id": feedback_id},
            priority=40,
            status="pending",
            idempotency_key=f"{target_id}::review::full::pin::{feedback_id}",
        ))
        if inserted:
            enqueued += 1
            tasks.append("reflection")

    if h.elo is None:
        inserted = await task_repo.enqueue(conn, Task(
            id=ids.task_id(),
            session_id=session_id,
            created_at=datetime.now(UTC),
            agent="ranking",
            action="AddToTournament",
            target_id=target_id,
            payload={"reason": "human_pin", "feedback_id": feedback_id},
            priority=70,
            status="pending",
            idempotency_key=f"{target_id}::ranking::add::pin::{feedback_id}",
        ))
        if inserted:
            enqueued += 1
            tasks.append("ranking_add")

    if has_complete_review and h.elo is not None:
        inserted = await task_repo.enqueue(conn, Task(
            id=ids.task_id(),
            session_id=session_id,
            created_at=datetime.now(UTC),
            agent="ranking",
            action="RunTournamentBatch",
            target_id=None,
            payload={"focus": target_id, "reason": "human_pin", "feedback_id": feedback_id},
            priority=60,
            status="pending",
            idempotency_key=f"{target_id}::ranking::focus::pin::{feedback_id}",
        ))
        if inserted:
            enqueued += 1
            tasks.append("ranking_focus")

        inserted = await task_repo.enqueue(conn, Task(
            id=ids.task_id(),
            session_id=session_id,
            created_at=datetime.now(UTC),
            agent="evolution",
            action="EvolveTopHypotheses",
            target_id=target_id,
            payload={
                "focus": target_id,
                "top_k": 6,
                "strategies": ["simplify", "feasibility", "out_of_box"],
                "reason": "human_pin",
                "feedback_id": feedback_id,
            },
            priority=90,
            status="pending",
            idempotency_key=f"{target_id}::evolution::pin::{feedback_id}",
        ))
        if inserted:
            enqueued += 1
            tasks.append("evolution_focus")

    return {
        "enqueued": enqueued,
        "tasks": tasks,
        "state": "pinned",
        "has_complete_review": has_complete_review,
    }
