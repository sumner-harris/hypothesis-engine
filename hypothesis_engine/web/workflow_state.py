"""Workflow state helpers for the live web UI."""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from ..config import Config

WORKFLOW_STAGES: list[dict[str, str]] = [
    {
        "key": "generation",
        "label": "Generation",
        "caption": "new ideas",
    },
    {
        "key": "reflection",
        "label": "Reflection",
        "caption": "reviews",
    },
    {
        "key": "ranking",
        "label": "Ranking",
        "caption": "tournament",
    },
    {
        "key": "evolution",
        "label": "Evolution",
        "caption": "refinement",
    },
    {
        "key": "proximity",
        "label": "Proximity",
        "caption": "clusters",
    },
    {
        "key": "metareview",
        "label": "Meta-review",
        "caption": "synthesis",
    },
]

WORKFLOW_STAGE_KEYS = (*(stage["key"] for stage in WORKFLOW_STAGES), "literature_review")


def workflow_stage_for_task(agent: str | None, action: str | None = None) -> str | None:
    """Map task agent/action values onto diagram stages."""
    if agent in WORKFLOW_STAGE_KEYS:
        return agent
    if action in {"GenerateSystemFeedback", "GenerateFinalResearchOverview"}:
        return "metareview"
    return None


async def workflow_state(
    conn: aiosqlite.Connection,
    session_id: str,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Return JSON-ready active workflow state for a session."""
    async with conn.execute(
        """SELECT t.id, t.agent, t.action, t.status, t.target_id, t.started_at,
                  h.created_by AS target_created_by
             FROM tasks AS t
             LEFT JOIN hypotheses AS h ON h.id = t.target_id
            WHERE t.session_id=? AND t.status IN ('leased', 'in_progress')
            ORDER BY t.started_at DESC, t.created_at DESC""",
        (session_id,),
    ) as cur:
        rows = await cur.fetchall()

    counts = {key: 0 for key in WORKFLOW_STAGE_KEYS}
    active_tasks: list[dict[str, Any]] = []
    for row in rows:
        stage = workflow_stage_for_task(row["agent"], row["action"])
        if stage is None:
            continue
        counts[stage] += 1
        active_tasks.append(
            {
                "task_id": row["id"],
                "stage": stage,
                "agent": row["agent"],
                "action": row["action"],
                "status": row["status"],
                "target_id": row["target_id"],
                "target_created_by": row["target_created_by"],
                "started_at": row["started_at"],
            }
        )

    literature_review_active, literature_review_sources = _active_literature_review_calls(cfg, session_id)
    if literature_review_active:
        counts["literature_review"] = literature_review_active

    return {
        "stages": WORKFLOW_STAGES,
        "counts": counts,
        "active": [key for key, count in counts.items() if count > 0],
        "active_tasks": active_tasks,
        "active_literature_review_sources": sorted(literature_review_sources),
    }


def _active_literature_review_calls(cfg: Config | None, session_id: str) -> tuple[int, set[str]]:
    if cfg is None:
        return 0, set()
    root = cfg.session_artifact_dir(session_id) / "literature_review" / "active_calls"
    if not root.is_dir():
        return 0, set()
    now = datetime.now(UTC).timestamp()
    active = 0
    sources: set[str] = set()
    for path in root.glob("*.json"):
        try:
            age_seconds = now - path.stat().st_mtime
        except OSError:
            continue
        if age_seconds > 900:
            with suppress(OSError):
                path.unlink(missing_ok=True)
            continue
        active += 1
        source = _literature_review_marker_source(path)
        if source:
            sources.add(source)
    return active, sources


def _literature_review_marker_source(path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    source = str(data.get("trigger_agent") or "").strip()
    return source if source in {"generation", "reflection", "evolution"} else None
