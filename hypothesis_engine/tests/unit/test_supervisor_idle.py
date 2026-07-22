"""Tests for Supervisor idle scheduling edge cases."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from hypothesis_engine import ids
from hypothesis_engine.agents.supervisor import (
    Supervisor,
    _drain_finished_inflight,
    _initial_generation_payload,
)
from hypothesis_engine.config import Config
from hypothesis_engine.models import (
    Hypothesis,
    ResearchPlan,
    Review,
    ReviewScores,
    Session,
    SystemFeedback,
    Task,
    TaskResult,
    TournamentMatch,
)
from hypothesis_engine.orchestrator.termination import EloSnapshot, StabilityTracker
from hypothesis_engine.storage.repos import feedback as fb_repo
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import reviews as rev_repo
from hypothesis_engine.storage.repos import sessions as sess_repo
from hypothesis_engine.storage.repos import tasks as task_repo
from hypothesis_engine.storage.repos import tournaments as tourney_repo
from hypothesis_engine.tools import rag as rag_mod


def _session() -> Session:
    now = datetime.now(UTC)
    return Session(
        id="ses_idle",
        created_at=now,
        updated_at=now,
        status="running",
        research_goal="g",
        research_plan=ResearchPlan(objective="g"),
        config_snapshot={},
        budget_tokens=1_000_000,
        budget_usd=10.0,
        wall_deadline=now + timedelta(hours=1),
    )


def _hypothesis(hid: str, session_id: str) -> Hypothesis:
    return Hypothesis(
        id=hid,
        session_id=session_id,
        created_at=datetime.now(UTC),
        created_by="generation",
        strategy="literature",
        title=hid,
        summary="summary",
        full_text="full text",
        artifact_path=f"artifacts/{hid}.md",
        elo=1200.0,
        matches_played=1,
        state="in_tournament",
    )


async def _insert_matches(
    conn,
    session_id: str,
    n: int,
    *,
    start: int = 0,
    hyp_a: str = "hyp_mature_0",
    hyp_b: str = "hyp_mature_1",
) -> None:
    for i in range(start, start + n):
        await tourney_repo.insert_match(
            conn,
            TournamentMatch(
                id=f"mat_idle_{i}",
                session_id=session_id,
                created_at=datetime.now(UTC),
                hyp_a=hyp_a,
                hyp_b=hyp_b,
                mode="pairwise",
                winner="a",
                elo_a_before=1200.0,
                elo_b_before=1200.0,
                elo_a_after=1216.0,
                elo_b_after=1184.0,
                rationale="rationale",
            ),
        )


def _stable_tracker(
    *,
    match_count: int = 0,
    pool_size: int = 5,
    top_ids: tuple[str, ...] = ("hyp_mature_4", "hyp_mature_3", "hyp_mature_2"),
    top_elos: tuple[float, ...] = (1204.0, 1203.0, 1202.0),
) -> StabilityTracker:
    tracker = StabilityTracker(k=len(top_ids), n=3, eps=25.0)
    for i in range(3):
        tracker.push(
            EloSnapshot(
                match_count=max(0, match_count - 2 + i),
                top_ids=top_ids,
                top_elos=tuple(elo + i for elo in top_elos),
                pool_size=pool_size,
            )
        )
    return tracker


async def test_drain_finished_inflight_removes_completed_tasks() -> None:
    completed = asyncio.create_task(asyncio.sleep(0))
    pending = asyncio.create_task(asyncio.sleep(60))
    await completed
    inflight = {completed, pending}

    drained = _drain_finished_inflight(inflight)

    assert drained == 1
    assert completed not in inflight
    assert pending in inflight
    pending.cancel()
    with suppress(asyncio.CancelledError):
        await pending


def test_initial_generation_payload_assigns_distinct_discovery_profiles() -> None:
    payloads = [_initial_generation_payload(i, 4) for i in range(4)]

    assert [payload["initial_index"] for payload in payloads] == [0, 1, 2, 3]
    assert {payload["initial_total"] for payload in payloads} == {4}
    assert [payload["discovery_profile"]["id"] for payload in payloads] == [
        "mechanism",
        "synthesis_route",
        "characterization",
        "theory_modeling",
    ]
    assert all(payload["discovery_group"] == "initial" for payload in payloads)


def test_initial_generation_payload_uses_configured_discovery_profiles(tmp_path) -> None:
    profile_path = tmp_path / "custom_profiles.yaml"
    profile_path.write_text(
        """profiles:
  - id: field_a
    label: Field A lens
    objective: Search one field-specific literature slice.
  - id: field_b
    label: Field B lens
    objective: Search a second field-specific literature slice.
""",
        encoding="utf-8",
    )
    cfg = Config()
    cfg.generation.discovery_profiles = str(profile_path)

    payloads = [_initial_generation_payload(i, 3, cfg) for i in range(3)]

    assert [payload["discovery_profile"]["id"] for payload in payloads] == [
        "field_a",
        "field_b",
        "field_a",
    ]
    assert {payload["initial_total"] for payload in payloads} == {3}


async def test_decide_next_steps_returns_zero_when_idle_task_key_already_used(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(conn, _hypothesis("hyp_a", session.id))
    await hyp_repo.insert(conn, _hypothesis("hyp_b", session.id))

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 1
    tasks = await task_repo.count_by_status(conn, session.id)
    assert tasks == {"pending": 1}

    async with conn.execute(
        "SELECT id FROM tasks WHERE session_id=? AND status=?",
        (session.id, "pending"),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    await task_repo.complete(conn, row["id"])

    assert await supervisor._decide_next_steps(conn, session) == 0
    assert await task_repo.count_by_status(conn, session.id) == {"done": 1}


async def test_enqueue_missing_reflections_for_evolved_drafts(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)

    evolved = _hypothesis("hyp_evolved", session.id)
    evolved.created_by = "evolution"
    evolved.strategy = "simplify"
    evolved.state = "draft"
    evolved.elo = None
    evolved.matches_played = 0
    await hyp_repo.insert(conn, evolved)

    already_queued = _hypothesis("hyp_already_queued", session.id)
    already_queued.created_by = "evolution"
    already_queued.strategy = "combine"
    already_queued.state = "draft"
    already_queued.elo = None
    already_queued.matches_played = 0
    await hyp_repo.insert(conn, already_queued)
    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="reflection",
            action="ReviewHypothesis",
            target_id=already_queued.id,
            payload={"kind": "full"},
            priority=100,
            status="pending",
            idempotency_key=f"{already_queued.id}::review::full",
        ),
    )

    already_reviewed = _hypothesis("hyp_already_reviewed", session.id)
    already_reviewed.created_by = "evolution"
    already_reviewed.strategy = "out_of_box"
    already_reviewed.state = "draft"
    already_reviewed.elo = None
    already_reviewed.matches_played = 0
    await hyp_repo.insert(conn, already_reviewed)
    await rev_repo.insert(
        conn,
        Review(
            id=ids.review_id(already_reviewed.id, "full", iteration=0),
            hypothesis_id=already_reviewed.id,
            session_id=session.id,
            created_at=datetime.now(UTC),
            kind="full",
            verdict="neutral",
            scores=ReviewScores(novelty=0.5, correctness=0.5, testability=0.5, feasibility=0.5),
            body="complete",
            artifact_path="artifacts/review.json",
        ),
    )

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._enqueue_missing_reflections_for_drafts(conn, session.id) == 1
    assert await supervisor._enqueue_missing_reflections_for_drafts(conn, session.id) == 0

    async with conn.execute(
        """SELECT target_id, idempotency_key
             FROM tasks
            WHERE session_id=? AND agent='reflection'
            ORDER BY created_at""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert [(r["target_id"], r["idempotency_key"]) for r in rows] == [
        (already_queued.id, f"{already_queued.id}::review::full"),
        (evolved.id, f"{evolved.id}::review::full"),
    ]


async def test_decide_next_steps_reprocesses_active_pin_feedback(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    pinned = _hypothesis("hyp_pinned", session.id)
    pinned.state = "pinned"
    pinned.elo = 1216.0
    other = _hypothesis("hyp_other", session.id)
    await hyp_repo.insert(conn, pinned)
    await hyp_repo.insert(conn, other)
    await rev_repo.insert(
        conn,
        Review(
            id=ids.review_id(pinned.id, "full", iteration=0),
            hypothesis_id=pinned.id,
            session_id=session.id,
            created_at=datetime.now(UTC),
            kind="full",
            verdict="missing_piece",
            scores=ReviewScores(novelty=0.7, correctness=0.6, testability=0.8, feasibility=0.5),
            body="complete",
            artifact_path="artifacts/review.json",
        ),
    )
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_pin",
            session_id=session.id,
            created_at=datetime.now(UTC),
            source="human",
            kind="pin",
            target_id=pinned.id,
            text="Explore this further",
            active=True,
        ),
    )

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 4
    async with conn.execute(
        "SELECT agent, action, target_id, payload FROM tasks WHERE session_id=? ORDER BY priority",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert [(r["agent"], r["action"]) for r in rows[:3]] == [
        ("metareview", "GenerateSystemFeedback"),
        ("ranking", "RunTournamentBatch"),
        ("evolution", "EvolveTopHypotheses"),
    ]
    assert rows[0]["target_id"] is None
    assert '"feedback_id": "fb_pin"' in rows[0]["payload"]
    assert rows[1]["target_id"] is None
    assert '"focus": "hyp_pinned"' in rows[1]["payload"]
    assert rows[2]["target_id"] == "hyp_pinned"

    assert await supervisor._decide_next_steps(conn, session) == 0


async def test_decide_next_steps_enqueues_parallel_ranking_warmup_before_evolution_gate(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 4
    tmp_cfg.evolution.min_mature = 5

    for i in range(3):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 3
        h.elo = 1300.0 - i
        await hyp_repo.insert(conn, h)
    for i, matches in enumerate([2, 1, 1, 0, 0]):
        h = _hypothesis(f"hyp_warmup_{i}", session.id)
        h.matches_played = matches
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 4
    async with conn.execute(
        "SELECT agent, action, payload FROM tasks WHERE session_id=? ORDER BY created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert all(r["agent"] == "ranking" and r["action"] == "RunTournamentBatch" for r in rows)
    payloads = [json.loads(r["payload"]) for r in rows]
    assert [p["focus"] for p in payloads] == [
        "hyp_warmup_0",
        "hyp_warmup_2",
        "hyp_warmup_1",
        "hyp_warmup_4",
    ]
    assert all(p["reason"] == "warmup" for p in payloads)

    assert await supervisor._decide_next_steps(conn, session) == 0


async def test_decide_next_steps_throttles_small_warm_pool_when_feed_backlog_exists(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.small_pool_backlog_size = 6
    tmp_cfg.ranking.small_pool_backlog_match_cap = 5

    for i in range(3):
        h = _hypothesis(f"hyp_warm_{i}", session.id)
        h.matches_played = 5
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="reflection",
            action="ReviewHypothesis",
            target_id="hyp_incoming",
            payload={"kind": "full"},
            priority=100,
            status="in_progress",
        ),
    )
    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 0

    await conn.execute(
        "UPDATE tasks SET status='done' WHERE session_id=? AND agent='reflection'",
        (session.id,),
    )
    await conn.commit()

    assert await supervisor._decide_next_steps(conn, session) == 1
    async with conn.execute(
        "SELECT agent, action FROM tasks WHERE session_id=? AND agent='ranking'",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()
    assert [(r["agent"], r["action"]) for r in rows] == [("ranking", "RunTournamentBatch")]


async def test_decide_next_steps_enqueues_parallel_evolution_when_pool_is_mature(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 3
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 4
    tmp_cfg.evolution.top_k = 8

    for i in range(5):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 3
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session, tracker=_stable_tracker()) == 7
    async with conn.execute(
        "SELECT agent, action, target_id, payload FROM tasks WHERE session_id=? ORDER BY priority, created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    evolution_rows = [r for r in rows if r["agent"] == "evolution"]
    assert len(evolution_rows) == 4
    payloads = [json.loads(r["payload"]) for r in evolution_rows]
    assert payloads[0]["strategies"] == ["combine"]
    focused = [p for p in payloads if p.get("focus")]
    assert len(focused) == 3
    assert all(p["strategies"] == ["simplify", "feasibility", "out_of_box"] for p in focused)
    assert [r["target_id"] for r in evolution_rows[1:]] == [p["focus"] for p in focused]
    ranking_rows = [r for r in rows if r["agent"] == "ranking"]
    assert len(ranking_rows) == 3
    ranking_payloads = [json.loads(r["payload"]) for r in ranking_rows]
    assert [p["reason"] for p in ranking_payloads] == [
        "top_refine",
        "refine",
        "refine",
    ]
    assert [p["focus"] for p in ranking_payloads] == [
        "hyp_mature_4",
        "hyp_mature_3",
        "hyp_mature_2",
    ]

    assert await supervisor._decide_next_steps(conn, session, tracker=_stable_tracker()) == 0


async def test_decide_next_steps_does_not_enqueue_idle_evolution_without_rank_stability(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 2
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 4

    for i in range(5):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 3
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 2
    async with conn.execute(
        "SELECT agent FROM tasks WHERE session_id=? ORDER BY created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert [r["agent"] for r in rows] == ["ranking", "ranking"]


async def test_decide_next_steps_does_not_enqueue_idle_evolution_when_active(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 2
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 4

    for i in range(5):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 3
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="evolution",
            action="EvolveTopHypotheses",
            payload={"top_k": 5, "strategies": ["combine"]},
            priority=140,
            status="in_progress",
            idempotency_key=f"{session.id}::evolution::idle::0::combine",
        ),
    )

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session, tracker=_stable_tracker()) == 2
    async with conn.execute(
        "SELECT agent, status FROM tasks WHERE session_id=? ORDER BY created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert sum(1 for r in rows if r["agent"] == "evolution") == 1
    assert sum(1 for r in rows if r["agent"] == "ranking") == 2


async def test_decide_next_steps_throttles_idle_evolution_by_match_cadence(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 2
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 2
    tmp_cfg.evolution.min_tournament_matches = 10
    tmp_cfg.evolution.min_matches_between_batches = 5

    for i in range(5):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 5
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="evolution",
            action="EvolveTopHypotheses",
            payload={"top_k": 5, "strategies": ["combine"]},
            priority=140,
            status="done",
            idempotency_key=f"{session.id}::evolution::idle::10::combine",
        ),
    )
    await _insert_matches(conn, session.id, 12)

    supervisor = Supervisor(tmp_cfg)

    tracker = _stable_tracker(match_count=15)

    assert await supervisor._decide_next_steps(conn, session, tracker=tracker) == 2
    async with conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE session_id=? AND agent='evolution'",
        (session.id,),
    ) as cur:
        row = await cur.fetchone()
    assert row["n"] == 1

    await _insert_matches(conn, session.id, 3, start=12)

    assert await supervisor._decide_next_steps(conn, session, tracker=tracker) == 4
    async with conn.execute(
        """SELECT idempotency_key
              FROM tasks
             WHERE session_id=? AND agent='evolution'
             ORDER BY created_at""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert [r["idempotency_key"] for r in rows] == [
        f"{session.id}::evolution::idle::10::combine",
        f"{session.id}::evolution::idle::15::combine",
        f"{session.id}::evolution::idle::15::focus::hyp_mature_4",
    ]


async def test_decide_next_steps_does_not_enqueue_evolution_at_max_ideas(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.run.max_ideas = 5
    tmp_cfg.ranking.idle_parallel_tasks = 3
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 4

    for i in range(5):
        h = _hypothesis(f"hyp_mature_{i}", session.id)
        h.matches_played = 3
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 3
    async with conn.execute(
        "SELECT agent, action, payload FROM tasks WHERE session_id=? ORDER BY priority, created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert all(r["agent"] == "ranking" for r in rows)
    payloads = [json.loads(r["payload"]) for r in rows]
    assert payloads[0]["reason"] == "top_refine"
    assert all(p["reason"] in {"top_refine", "refine"} for p in payloads)


async def test_decide_next_steps_keeps_top_ranked_hypotheses_challenged(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 6
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 0

    top_a = _hypothesis("hyp_top_a", session.id)
    top_a.matches_played = 8
    top_a.elo = 1300.0
    top_b = _hypothesis("hyp_top_b", session.id)
    top_b.matches_played = 7
    top_b.elo = 1290.0
    await hyp_repo.insert(conn, top_a)
    await hyp_repo.insert(conn, top_b)

    for i in range(8):
        h = _hypothesis(f"hyp_under_{i}", session.id)
        h.matches_played = 3
        h.elo = 1200.0 + i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 6
    async with conn.execute(
        "SELECT priority, payload FROM tasks WHERE session_id=? AND agent='ranking' ORDER BY priority, created_at",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    payloads = [json.loads(r["payload"]) for r in rows]
    assert [p["focus"] for p in payloads[:2]] == ["hyp_top_a", "hyp_top_b"]
    assert [p["reason"] for p in payloads[:2]] == ["top_refine", "top_refine"]
    assert [r["priority"] for r in rows[:2]] == [98, 98]
    assert all(p["reason"] == "refine" for p in payloads[2:])


async def test_decide_next_steps_ignores_legacy_individual_match_cap(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.run.max_matches_per_idea = 1
    tmp_cfg.ranking.idle_parallel_tasks = 3
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.evolution.idle_parallel_tasks = 0

    for i in range(5):
        h = _hypothesis(f"hyp_over_cap_{i}", session.id)
        h.matches_played = 30 + i
        h.elo = 1300.0 - i
        await hyp_repo.insert(conn, h)

    supervisor = Supervisor(tmp_cfg)

    assert await supervisor._decide_next_steps(conn, session) == 3
    async with conn.execute(
        """SELECT payload
              FROM tasks
             WHERE session_id=? AND agent='ranking'
             ORDER BY priority, created_at""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    payloads = [json.loads(r["payload"]) for r in rows]
    assert [p["focus"] for p in payloads] == [
        "hyp_over_cap_0",
        "hyp_over_cap_1",
        "hyp_over_cap_2",
    ]
    assert {p["reason"] for p in payloads} <= {"top_refine", "refine"}


async def test_tournament_match_followup_enqueues_top_refine_despite_backlog(conn, tmp_cfg) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.ranking.idle_parallel_tasks = 6
    tmp_cfg.evolution.min_mature = 3
    tmp_cfg.vectors.full_recluster_every_matches = 999

    top_a = _hypothesis("hyp_top_a", session.id)
    top_a.matches_played = 8
    top_a.elo = 1300.0
    top_b = _hypothesis("hyp_top_b", session.id)
    top_b.matches_played = 7
    top_b.elo = 1290.0
    top_c = _hypothesis("hyp_top_c", session.id)
    top_c.matches_played = 6
    top_c.elo = 1280.0
    for h in [top_a, top_b, top_c]:
        await hyp_repo.insert(conn, h)

    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="ranking",
            action="RunTournamentBatch",
            payload={"focus": "hyp_backlog"},
            priority=120,
            status="pending",
        ),
    )

    for i in range(6):
        await tourney_repo.insert_match(
            conn,
            TournamentMatch(
                id=f"mat_top_refine_{i}",
                session_id=session.id,
                created_at=datetime.now(UTC),
                hyp_a=top_a.id,
                hyp_b=top_c.id,
                mode="pairwise",
                winner="a",
                elo_a_before=1200.0,
                elo_b_before=1200.0,
                elo_a_after=1216.0,
                elo_b_after=1184.0,
                rationale="rationale",
            ),
        )

    supervisor = Supervisor(tmp_cfg)
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="ranking",
        action="RunTournamentBatch",
        payload={},
        priority=120,
        status="done",
    )

    await supervisor._apply_follow_ups(
        conn, session, task, TaskResult(kind="tournament_match_complete")
    )
    await supervisor._apply_follow_ups(
        conn, session, task, TaskResult(kind="tournament_match_complete")
    )

    async with conn.execute(
        """SELECT priority, payload, idempotency_key
             FROM tasks
            WHERE session_id=? AND agent='ranking'
            ORDER BY priority, created_at""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    payloads = [json.loads(r["payload"]) for r in rows]
    top_refine = [p for p in payloads if p.get("reason") == "top_refine"]
    assert [p["focus"] for p in top_refine] == ["hyp_top_a", "hyp_top_b"]
    assert [r["priority"] for r in rows[:2]] == [98, 98]
    assert len(top_refine) == 2
    assert rows[0]["idempotency_key"].endswith("::top_refine::1::hyp_top_a")
    assert rows[1]["idempotency_key"].endswith("::top_refine::1::hyp_top_b")
    assert payloads[-1] == {"focus": "hyp_backlog"}


async def test_tournament_match_followup_enqueues_due_metareview_without_idle_queue(
    conn, tmp_cfg
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.vectors.full_recluster_every_matches = 999

    hyp_a = _hypothesis("hyp_a", session.id)
    hyp_b = _hypothesis("hyp_b", session.id)
    hyp_a.matches_played = 50
    hyp_b.matches_played = 50
    await hyp_repo.insert(conn, hyp_a)
    await hyp_repo.insert(conn, hyp_b)
    for i in range(50):
        await tourney_repo.insert_match(
            conn,
            TournamentMatch(
                id=f"mat_{i}",
                session_id=session.id,
                created_at=datetime.now(UTC),
                hyp_a=hyp_a.id,
                hyp_b=hyp_b.id,
                mode="pairwise",
                winner="a",
                elo_a_before=1200.0,
                elo_b_before=1200.0,
                elo_a_after=1216.0,
                elo_b_after=1184.0,
                rationale="rationale",
            ),
        )

    supervisor = Supervisor(tmp_cfg)
    task = Task(
        id=ids.task_id(),
        session_id=session.id,
        created_at=datetime.now(UTC),
        agent="ranking",
        action="RunTournamentBatch",
        payload={},
        priority=120,
        status="done",
    )
    result = TaskResult(kind="tournament_match_complete")

    await supervisor._apply_follow_ups(conn, session, task, result)
    await supervisor._apply_follow_ups(conn, session, task, result)

    async with conn.execute(
        """SELECT agent, action, priority, payload, idempotency_key
              FROM tasks
             WHERE session_id=? AND agent='metareview'
             ORDER BY created_at""",
        (session.id,),
    ) as cur:
        rows = await cur.fetchall()

    assert len(rows) == 1
    assert (rows[0]["agent"], rows[0]["action"]) == (
        "metareview",
        "GenerateSystemFeedback",
    )
    assert rows[0]["priority"] == 95
    payload = json.loads(rows[0]["payload"])
    assert payload == {
        "reason": "periodic_tournament_threshold",
        "match_count": 50,
        "feedback_index": 1,
    }
    assert rows[0]["idempotency_key"] == f"{session.id}::metareview::feedback::1"


async def test_supervisor_waits_for_rag_ingest_before_non_generation_work(
    conn, tmp_cfg, monkeypatch, tmp_path
) -> None:
    from hypothesis_engine.tools.base import ToolCtx

    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.auto_ingest_arxiv_pdfs = True
    tmp_cfg.rag.package_path = str(tmp_path)

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_ingest(_ctx, _records):
        started.set()
        await release.wait()
        return {"enabled": True, "ingested": 1}

    monkeypatch.setattr(rag_mod, "ingest_arxiv_records", fake_ingest)
    rag_mod.schedule_arxiv_ingest(
        ToolCtx(cfg=tmp_cfg, session_id=session.id),
        [{"pdf_url": "https://arxiv.org/pdf/2401.00001v1", "title": "paper"}],
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        supervisor = Supervisor(tmp_cfg)

        assert await supervisor._rag_ingest_wait_needed(conn, session.id) is True

        await task_repo.enqueue(
            conn,
            Task(
                id=ids.task_id(),
                session_id=session.id,
                created_at=datetime.now(UTC),
                agent="generation",
                action="CreateInitialHypotheses",
                payload={},
                priority=100,
                status="pending",
            ),
        )
        assert await supervisor._rag_ingest_wait_needed(conn, session.id) is False

        await task_repo.enqueue(
            conn,
            Task(
                id=ids.task_id(),
                session_id=session.id,
                created_at=datetime.now(UTC),
                agent="reflection",
                action="ReviewHypothesis",
                payload={},
                priority=100,
                status="pending",
            ),
        )
        assert await supervisor._rag_ingest_wait_needed(conn, session.id) is True

        wait_task = asyncio.create_task(supervisor._wait_for_rag_ingest_if_needed(conn, session.id))
        await asyncio.sleep(0.05)
        assert wait_task.done() is False

        release.set()
        assert await asyncio.wait_for(wait_task, timeout=1.0) is True
        assert rag_mod.background_ingest_pending(tmp_cfg, session.id) is False
    finally:
        release.set()
        await rag_mod.wait_for_background_ingest_step(tmp_cfg, session.id, timeout_seconds=1.0)


async def test_supervisor_rag_wait_releases_when_seed_kb_is_ready(
    conn, tmp_cfg, monkeypatch
) -> None:
    session = _session()
    await sess_repo.insert(conn, session)
    tmp_cfg.rag.enabled = True
    tmp_cfg.rag.generation_wait_min_indexed_papers = 30
    calls = 0

    monkeypatch.setattr(rag_mod, "background_ingest_pending", lambda _cfg, _session_id: True)
    monkeypatch.setattr(
        rag_mod,
        "background_ingest_status",
        lambda _cfg, _session_id: {
            "enabled": True,
            "pending_background_ingest": True,
            "seed_chunk_count": 1241,
            "seed_kb_ready": True,
        },
    )

    async def fake_wait_step(_cfg, _session_id, *, timeout_seconds):
        nonlocal calls
        calls += 1
        assert timeout_seconds == 1.0
        return {
            "enabled": True,
            "pending_background_ingest": True,
            "seed_chunk_count": 1241,
            "seed_kb_ready": True,
        }

    monkeypatch.setattr(rag_mod, "wait_for_background_ingest_step", fake_wait_step)
    await task_repo.enqueue(
        conn,
        Task(
            id=ids.task_id(),
            session_id=session.id,
            created_at=datetime.now(UTC),
            agent="reflection",
            action="ReviewHypothesis",
            payload={},
            priority=100,
            status="pending",
        ),
    )

    supervisor = Supervisor(tmp_cfg)
    assert await supervisor._wait_for_rag_ingest_if_needed(conn, session.id) is True
    assert calls == 1
