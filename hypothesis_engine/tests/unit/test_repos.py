# Modified from the original work.
"""Roundtrip tests for the storage repositories."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hypothesis_engine import ids
from hypothesis_engine.models import (
    CitedPaper,
    Hypothesis,
    ResearchPlan,
    Review,
    ReviewScores,
    Session,
    SystemFeedback,
    Task,
    TournamentMatch,
)
from hypothesis_engine.orchestrator.feedback_actions import apply_human_feedback_actions
from hypothesis_engine.storage.repos import (
    feedback as fb_repo,
)
from hypothesis_engine.storage.repos import (
    hypotheses as hyp_repo,
)
from hypothesis_engine.storage.repos import (
    reviews as rev_repo,
)
from hypothesis_engine.storage.repos import (
    sessions as sess_repo,
)
from hypothesis_engine.storage.repos import (
    tasks as task_repo,
)
from hypothesis_engine.storage.repos import (
    tournaments as tourney_repo,
)


def _now() -> datetime:
    return datetime.now(UTC)


async def _make_session(conn, sid: str = "ses_test") -> Session:
    s = Session(
        id=sid, created_at=_now(), updated_at=_now(), status="running",
        research_goal="Test goal",
        research_plan=ResearchPlan(objective="x", preferences=["specificity"]),
        config_snapshot={"k": 1}, budget_tokens=10000, budget_usd=1.0,
    )
    await sess_repo.insert(conn, s)
    return s


@pytest.mark.asyncio
async def test_sessions_roundtrip(conn) -> None:
    s = await _make_session(conn)
    s2 = await sess_repo.fetch(conn, s.id)
    assert s2 is not None
    assert s2.research_goal == s.research_goal
    assert s2.research_plan.objective == "x"
    assert s2.research_plan.preferences == ["specificity"]
    assert s2.config_snapshot == {"k": 1}


@pytest.mark.asyncio
async def test_hypothesis_insert_or_ignore_dedupes_on_deterministic_id(conn) -> None:
    s = await _make_session(conn)
    statement = "Hypothesis: X causes Y via Z."
    hid = ids.hypothesis_id(s.id, "generation/literature", statement)
    h = Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="t", summary=statement, full_text="long",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
        state="draft",
    )
    assert await hyp_repo.insert(conn, h) is True
    # Same statement → same id → INSERT OR IGNORE returns False
    h2 = Hypothesis(**{**h.model_dump(), "title": "different title"})
    assert await hyp_repo.insert(conn, h2) is False
    fetched = await hyp_repo.fetch(conn, hid)
    assert fetched is not None
    assert fetched.title == "t"  # original wins


@pytest.mark.asyncio
async def test_hypothesis_citations_roundtrip(conn) -> None:
    s = await _make_session(conn, sid="ses_citations")
    hid = ids.hypothesis_id(s.id, "generation/literature", "h cites source")
    h = Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="cited", summary="h cites source", full_text="long",
        citations=[CitedPaper(
            title="Important paper",
            url="https://example.test/paper",
            excerpt="Short supporting excerpt.",
            doi="10.1234/example",
            year=2024,
        )],
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
        state="draft",
    )

    assert await hyp_repo.insert(conn, h) is True
    fetched = await hyp_repo.fetch(conn, hid)

    assert fetched is not None
    assert len(fetched.citations) == 1
    citation = fetched.citations[0]
    assert citation.title == "Important paper"
    assert citation.url == "https://example.test/paper"
    assert citation.excerpt == "Short supporting excerpt."
    assert citation.doi == "10.1234/example"
    assert citation.year == 2024


@pytest.mark.asyncio
async def test_init_tournament_only_runs_once(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "h")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json", state="reviewed",
    ))
    assert await hyp_repo.init_tournament(conn, hid, initial_elo=1200) is True
    # second call must be a no-op (Elo already set)
    assert await hyp_repo.init_tournament(conn, hid, initial_elo=9999) is False
    h = await hyp_repo.fetch(conn, hid)
    assert h is not None and h.elo == 1200


async def _claimed_ranking_task(conn, session_id: str, task_id: str) -> Task:
    await task_repo.enqueue(conn, Task(
        id=task_id, session_id=session_id,
        created_at=_now(), agent="ranking", action="RunTournamentBatch",
        target_id=None, payload={}, priority=100, status="pending",
    ))
    task = await task_repo.claim_one(conn, session_id, worker_id=task_id, lease_seconds=60)
    assert task is not None
    await task_repo.mark_in_progress(conn, task.id)
    return task


@pytest.mark.asyncio
async def test_tournament_pair_reservation_blocks_live_pair(conn) -> None:
    session = await _make_session(conn)
    task_a = await _claimed_ranking_task(conn, session.id, "tsk_pair_a")
    task_b = await _claimed_ranking_task(conn, session.id, "tsk_pair_b")
    pair_key = "hyp_a::hyp_b"

    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_a.id
    ) is True
    assert await tourney_repo.active_pair_keys(conn, session.id) == {pair_key}
    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_b.id
    ) is False

    await tourney_repo.release_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_a.id
    )

    assert await tourney_repo.active_pair_keys(conn, session.id) == set()
    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_b.id
    ) is True


@pytest.mark.asyncio
async def test_tournament_pair_reservation_cleans_expired_task(conn) -> None:
    session = await _make_session(conn)
    task_a = await _claimed_ranking_task(conn, session.id, "tsk_pair_expired")
    task_b = await _claimed_ranking_task(conn, session.id, "tsk_pair_next")
    pair_key = "hyp_a::hyp_b"

    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_a.id
    ) is True
    await conn.execute("UPDATE tasks SET lease_expires_at=0 WHERE id=?", (task_a.id,))
    await conn.commit()

    assert await tourney_repo.active_pair_keys(conn, session.id) == set()
    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key=pair_key, task_id=task_b.id
    ) is True


@pytest.mark.asyncio
async def test_tournament_pair_best_of_three_status_and_orientation(conn) -> None:
    session = await _make_session(conn)
    for hid in ("hyp_a", "hyp_b"):
        await hyp_repo.insert(conn, Hypothesis(
            id=hid, session_id=session.id, created_at=_now(),
            created_by="generation", strategy="literature",
            title=hid, summary="s", full_text="f",
            artifact_path=f"artifacts/{session.id}/hypotheses/{hid}.json",
            elo=1200, matches_played=0, state="in_tournament",
        ))

    first_a, first_b = await tourney_repo.next_pair_orientation(
        conn, session.id, "hyp_a", "hyp_b"
    )
    await tourney_repo.insert_match(conn, TournamentMatch(
        id="match_pair_1", session_id=session.id, created_at=_now(),
        hyp_a=first_a, hyp_b=first_b, mode="debate", winner="a",
        elo_a_before=1200, elo_b_before=1200,
    ))

    status = await tourney_repo.pair_status(conn, session.id, "hyp_a", "hyp_b")
    assert status["closed"] is False
    assert status["valid_matches"] == 1
    assert status["winner_hyp_id"] is None
    assert await tourney_repo.next_pair_orientation(
        conn, session.id, "hyp_a", "hyp_b"
    ) == (first_b, first_a)

    second_a, second_b = first_b, first_a
    first_round_winner_hyp = first_a
    second_winner = "a" if second_a == first_round_winner_hyp else "b"
    await tourney_repo.insert_match(conn, TournamentMatch(
        id="match_pair_2", session_id=session.id, created_at=_now(),
        hyp_a=second_a, hyp_b=second_b, mode="debate", winner=second_winner,
        elo_a_before=1200, elo_b_before=1200,
    ))

    status = await tourney_repo.pair_status(conn, session.id, "hyp_a", "hyp_b")
    assert status["closed"] is True
    assert status["valid_matches"] == 2
    assert status["winner_hyp_id"] == first_round_winner_hyp
    assert await tourney_repo.closed_pair_keys(conn, session.id) == {"hyp_a::hyp_b"}
    assert await tourney_repo.eligible_pair_count(
        conn, session.id, ["hyp_a", "hyp_b"], exclude_active=False
    ) == 0


@pytest.mark.asyncio
async def test_tournament_reservation_rejects_closed_pair(conn) -> None:
    session = await _make_session(conn)
    for hid in ("hyp_a", "hyp_b"):
        await hyp_repo.insert(conn, Hypothesis(
            id=hid, session_id=session.id, created_at=_now(),
            created_by="generation", strategy="literature",
            title=hid, summary="s", full_text="f",
            artifact_path=f"artifacts/{session.id}/hypotheses/{hid}.json",
            elo=1200, matches_played=0, state="in_tournament",
        ))
    await tourney_repo.insert_match(conn, TournamentMatch(
        id="closed_1", session_id=session.id, created_at=_now(),
        hyp_a="hyp_a", hyp_b="hyp_b", mode="debate", winner="a",
        elo_a_before=1200, elo_b_before=1200,
    ))
    await tourney_repo.insert_match(conn, TournamentMatch(
        id="closed_2", session_id=session.id, created_at=_now(),
        hyp_a="hyp_b", hyp_b="hyp_a", mode="debate", winner="b",
        elo_a_before=1200, elo_b_before=1200,
    ))
    task = await _claimed_ranking_task(conn, session.id, "tsk_closed_pair")

    assert await tourney_repo.reserve_pair(
        conn, session_id=session.id, pair_key="hyp_a::hyp_b", task_id=task.id
    ) is False


@pytest.mark.asyncio
async def test_set_state_if_only_applies_when_expected(conn) -> None:
    """set_state_if must only transition from one of the expected source states."""
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "h-state")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json", state="draft",
    ))
    # draft → reviewed: allowed
    applied = await hyp_repo.set_state_if(
        conn, hid, new_state="reviewed", expected_states=("draft",)
    )
    assert applied is True

    # Promote past reflection into the tournament.
    await hyp_repo.init_tournament(conn, hid, initial_elo=1200)

    # Reflection re-fires: must NOT drag in_tournament → reviewed.
    applied2 = await hyp_repo.set_state_if(
        conn, hid, new_state="reviewed", expected_states=("draft",)
    )
    assert applied2 is False
    h = await hyp_repo.fetch(conn, hid)
    assert h is not None and h.state == "in_tournament"


@pytest.mark.asyncio
async def test_review_id_iteration_collision_blocked(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "h")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json", state="draft",
    ))
    rid = ids.review_id(hid, "full", iteration=0)
    r = Review(
        id=rid, hypothesis_id=hid, session_id=s.id, created_at=_now(),
        kind="full", verdict="missing_piece",
        scores=ReviewScores(novelty=0.7, correctness=0.6, testability=0.5),
        body="ok", artifact_path=f"artifacts/{s.id}/reviews/{rid}.json",
    )
    assert await rev_repo.insert(conn, r) is True
    assert await rev_repo.insert(conn, r) is False   # idempotent




@pytest.mark.asyncio
async def test_review_upsert_replaces_existing_review(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "upsert")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="t", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json", state="draft",
    ))
    rid = ids.review_id(hid, "full", iteration=0)
    sparse = Review(
        id=rid, hypothesis_id=hid, session_id=s.id, created_at=_now(),
        kind="full", verdict=None, scores=ReviewScores(),
        body="# Review", artifact_path=f"artifacts/{s.id}/reviews/{rid}.json",
    )
    complete = Review(
        id=rid, hypothesis_id=hid, session_id=s.id, created_at=_now(),
        kind="full", verdict="neutral",
        scores=ReviewScores(novelty=0.1, correctness=0.2, testability=0.3, feasibility=0.4),
        body="complete", artifact_path=f"artifacts/{s.id}/reviews/{rid}.json",
    )

    assert await rev_repo.upsert(conn, sparse) is True
    assert await rev_repo.upsert(conn, complete) is True

    fetched = await rev_repo.fetch(conn, rid)
    assert fetched is not None
    assert fetched.verdict == "neutral"
    assert fetched.scores.feasibility == 0.4
    assert fetched.body == "complete"




@pytest.mark.asyncio
async def test_tournament_match_persists_prompt_metadata(conn) -> None:
    s = await _make_session(conn)
    hyp_a = ids.hypothesis_id(s.id, "generation/literature", "match a")
    hyp_b = ids.hypothesis_id(s.id, "generation/literature", "match b")
    for hid, title in ((hyp_a, "a"), (hyp_b, "b")):
        await hyp_repo.insert(conn, Hypothesis(
            id=hid, session_id=s.id, created_at=_now(),
            created_by="generation", strategy="literature",
            title=title, summary="s", full_text="f",
            artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
            state="in_tournament", elo=1200.0,
        ))

    match = TournamentMatch(
        id=ids.match_id(hyp_a, hyp_b, "round"),
        session_id=s.id,
        created_at=_now(),
        hyp_a=hyp_a,
        hyp_b=hyp_b,
        mode="debate",
        winner="b",
        elo_a_before=1200.0,
        elo_b_before=1200.0,
        elo_a_after=1184.0,
        elo_b_after=1216.0,
        prompt1_hyp_id=hyp_b,
        prompt2_hyp_id=hyp_a,
        prompt1_side="b",
        prompt2_side="a",
        winner_prompt_position=1,
        prompt1_chars=3400,
        prompt2_chars=3300,
        prompt_order_key="abc123",
    )

    assert await tourney_repo.insert_match(conn, match) is True
    async with conn.execute(
        """SELECT prompt1_hyp_id, prompt2_hyp_id, prompt1_side, winner_prompt_position,
                  prompt1_chars, prompt_order_key
             FROM tournament_matches WHERE id=?""",
        (match.id,),
    ) as cur:
        row = await cur.fetchone()

    assert row["prompt1_hyp_id"] == hyp_b
    assert row["prompt2_hyp_id"] == hyp_a
    assert row["prompt1_side"] == "b"
    assert row["winner_prompt_position"] == 1
    assert row["prompt1_chars"] == 3400
    assert row["prompt_order_key"] == "abc123"


@pytest.mark.asyncio
async def test_init_tournament_preserves_pinned_state_and_candidates_include_pins(conn) -> None:
    s = await _make_session(conn)
    pinned_id = ids.hypothesis_id(s.id, "generation/literature", "pinned")
    other_id = ids.hypothesis_id(s.id, "generation/literature", "other")
    await hyp_repo.insert(conn, Hypothesis(
        id=pinned_id, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="pinned", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{pinned_id}.json", state="pinned",
    ))
    await hyp_repo.insert(conn, Hypothesis(
        id=other_id, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="other", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{other_id}.json",
        state="in_tournament", elo=1210.0,
    ))

    assert await hyp_repo.init_tournament(conn, pinned_id) is True
    pinned = await hyp_repo.fetch(conn, pinned_id)
    assert pinned is not None
    assert pinned.state == "pinned"
    assert pinned.elo == 1200.0

    candidates = await hyp_repo.tournament_candidates(conn, s.id)
    assert {h.id for h in candidates} == {pinned_id, other_id}


@pytest.mark.asyncio
async def test_pin_feedback_sets_pin_and_enqueues_exploration(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "pin target")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="pin target", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
        state="in_tournament", elo=1216.0, matches_played=1,
    ))
    rid = ids.review_id(hid, "full", iteration=0)
    await rev_repo.insert(conn, Review(
        id=rid, hypothesis_id=hid, session_id=s.id, created_at=_now(),
        kind="full", verdict="missing_piece",
        scores=ReviewScores(novelty=0.7, correctness=0.6, testability=0.8, feasibility=0.5),
        body="complete", artifact_path=f"artifacts/{s.id}/reviews/{rid}.json",
    ))

    actions = await apply_human_feedback_actions(
        conn, session_id=s.id, feedback_id="fb_pin", kind="pin", target_id=hid
    )

    assert actions["state"] == "pinned"
    assert actions["enqueued"] == 3
    assert set(actions["tasks"]) == {"metareview", "ranking_focus", "evolution_focus"}
    pinned = await hyp_repo.fetch(conn, hid)
    assert pinned is not None and pinned.state == "pinned"

    async with conn.execute(
        "SELECT agent, action, target_id, payload FROM tasks WHERE session_id=? ORDER BY priority",
        (s.id,),
    ) as cur:
        rows = await cur.fetchall()
    assert [(r["agent"], r["action"]) for r in rows] == [
        ("metareview", "GenerateSystemFeedback"),
        ("ranking", "RunTournamentBatch"),
        ("evolution", "EvolveTopHypotheses"),
    ]
    assert '"feedback_id": "fb_pin"' in rows[0]["payload"]
    assert '"focus": "' + hid + '"' in rows[1]["payload"]
    assert rows[2]["target_id"] == hid


@pytest.mark.asyncio
async def test_pin_feedback_without_complete_review_enqueues_review_first(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "pin draft")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="pin draft", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json", state="draft",
    ))

    actions = await apply_human_feedback_actions(
        conn, session_id=s.id, feedback_id="fb_pin", kind="pin", target_id=hid
    )

    assert actions["state"] == "pinned"
    assert actions["has_complete_review"] is False
    assert set(actions["tasks"]) == {"metareview", "reflection", "ranking_add"}
    async with conn.execute(
        "SELECT agent, action, target_id FROM tasks WHERE session_id=? ORDER BY priority",
        (s.id,),
    ) as cur:
        rows = await cur.fetchall()
    assert [(r["agent"], r["action"], r["target_id"]) for r in rows] == [
        ("metareview", "GenerateSystemFeedback", None),
        ("reflection", "ReviewHypothesis", hid),
        ("ranking", "AddToTournament", hid),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,targeted", [("directive", False), ("preference", True)])
async def test_directive_and_preference_feedback_enqueue_metareview(conn, kind, targeted) -> None:
    s = await _make_session(conn)
    hid = None
    if targeted:
        hid = ids.hypothesis_id(s.id, "generation/literature", "preference target")
        await hyp_repo.insert(conn, Hypothesis(
            id=hid, session_id=s.id, created_at=_now(),
            created_by="generation", strategy="literature",
            title="target", summary="s", full_text="f",
            artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
            state="draft",
        ))

    actions = await apply_human_feedback_actions(
        conn, session_id=s.id, feedback_id=f"fb_{kind}", kind=kind, target_id=hid
    )

    assert actions == {"enqueued": 1, "tasks": ["metareview"]}
    if hid:
        h = await hyp_repo.fetch(conn, hid)
        assert h is not None and h.state == "draft"
    async with conn.execute(
        "SELECT agent, action, target_id, payload FROM tasks WHERE session_id=?",
        (s.id,),
    ) as cur:
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert (rows[0]["agent"], rows[0]["action"], rows[0]["target_id"]) == (
        "metareview", "GenerateSystemFeedback", None
    )
    assert f'"feedback_kind": "{kind}"' in rows[0]["payload"]


@pytest.mark.asyncio
async def test_rejection_feedback_rejects_target_cancels_pending_and_enqueues_metareview(conn) -> None:
    s = await _make_session(conn)
    hid = ids.hypothesis_id(s.id, "generation/literature", "reject target")
    await hyp_repo.insert(conn, Hypothesis(
        id=hid, session_id=s.id, created_at=_now(),
        created_by="generation", strategy="literature",
        title="target", summary="s", full_text="f",
        artifact_path=f"artifacts/{s.id}/hypotheses/{hid}.json",
        state="draft",
    ))
    await task_repo.enqueue(conn, Task(
        id=ids.task_id(), session_id=s.id, created_at=_now(),
        agent="reflection", action="ReviewHypothesis", target_id=hid,
        payload={}, priority=50, status="pending",
        idempotency_key=f"{hid}::review::pending",
    ))
    await task_repo.enqueue(conn, Task(
        id=ids.task_id(), session_id=s.id, created_at=_now(),
        agent="ranking", action="RunTournamentBatch", target_id=None,
        payload={"focus": hid}, priority=40, status="pending",
        idempotency_key=f"{hid}::ranking::focus::pending",
    ))

    actions = await apply_human_feedback_actions(
        conn, session_id=s.id, feedback_id="fb_reject", kind="rejection", target_id=hid
    )

    assert actions["state"] == "rejected"
    assert actions["cancelled_pending"] == 2
    assert actions["enqueued"] == 1
    assert actions["tasks"] == ["metareview"]
    rejected = await hyp_repo.fetch(conn, hid)
    assert rejected is not None and rejected.state == "rejected"
    assert rejected.created_by == "generation"
    assert rejected.strategy == "literature"
    assert rejected.parent_ids == []
    async with conn.execute(
        "SELECT agent, action, target_id, status FROM tasks WHERE session_id=? ORDER BY priority",
        (s.id,),
    ) as cur:
        rows = await cur.fetchall()
    assert [(r["agent"], r["action"], r["target_id"], r["status"]) for r in rows] == [
        ("metareview", "GenerateSystemFeedback", None, "pending"),
        ("ranking", "RunTournamentBatch", None, "cancelled"),
        ("reflection", "ReviewHypothesis", hid, "cancelled"),
    ]


@pytest.mark.asyncio
async def test_task_queue_claim_and_idempotency(conn) -> None:
    s = await _make_session(conn)
    t = Task(
        id=ids.task_id(), session_id=s.id, created_at=_now(),
        agent="reflection", action="ReviewHypothesis", target_id="hyp_z",
        payload={}, priority=100, status="pending",
        idempotency_key="hyp_z::review::full",
    )
    assert await task_repo.enqueue(conn, t) is True
    assert await task_repo.enqueue(conn, t) is False   # idempotency_key collision

    claimed = await task_repo.claim_one(conn, s.id, "w1", lease_seconds=60)
    assert claimed is not None
    assert claimed.id == t.id
    assert claimed.status == "leased"

    # nothing else to claim
    again = await task_repo.claim_one(conn, s.id, "w1", lease_seconds=60)
    assert again is None


@pytest.mark.asyncio
async def test_feedback_targeting(conn) -> None:
    s = await _make_session(conn)
    await fb_repo.insert(conn, SystemFeedback(
        id=ids.feedback_id(), session_id=s.id, created_at=_now(),
        source="human", kind="directive", target_id=None,
        text="focus on insulin signaling", active=True,
    ))
    await fb_repo.insert(conn, SystemFeedback(
        id=ids.feedback_id(), session_id=s.id, created_at=_now(),
        source="human", kind="pin", target_id="hyp_keep_me",
        text="pinned", active=True,
    ))
    global_only = await fb_repo.active_for_session(conn, s.id)
    assert len(global_only) == 1
    assert global_only[0].kind == "directive"

    targeted = await fb_repo.active_for_session(conn, s.id, target_id="hyp_keep_me")
    assert {f.kind for f in targeted} == {"directive", "pin"}

    all_human = await fb_repo.active_human_for_session(conn, s.id)
    assert {f.kind for f in all_human} == {"directive", "pin"}
