# Modified from the original work.
"""Tests for the pure Elo math + idempotent persistence."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime

import pytest

from hypothesis_engine import ids
from hypothesis_engine.models import Hypothesis, TournamentMatch
from hypothesis_engine.orchestrator.elo import expected_score, k_factor, update_elo
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import tournaments as tourney_repo


def test_equal_ratings_expect_half() -> None:
    assert expected_score(1200, 1200) == pytest.approx(0.5)


def test_higher_rating_favored() -> None:
    assert expected_score(1500, 1200) > 0.8
    assert expected_score(1200, 1500) < 0.2


def test_logistic_scale_controls_expected_probability() -> None:
    assert expected_score(1220, 1200, logistic_scale=150.0) > expected_score(1220, 1200)
    assert expected_score(1200, 1220, logistic_scale=150.0) < expected_score(1200, 1220)



def test_k_factor_decays() -> None:
    assert k_factor(0) == 32
    assert k_factor(4) == 32
    assert k_factor(5) == 16
    assert k_factor(100) == 16


def test_update_zero_sum() -> None:
    u = update_elo(1200, 1200, "a", matches_played_min=0)
    # zero-sum
    assert math.isclose(u.elo_a_after + u.elo_b_after, 2400, abs_tol=1e-9)
    # winner gains, loser drops
    assert u.elo_a_after > 1200
    assert u.elo_b_after < 1200


def test_underdog_win_is_high_payoff() -> None:
    u = update_elo(1100, 1500, "a", matches_played_min=10)
    # K=16, expected_a ~0.09, delta = 16*(1 - 0.09) = ~14.5
    assert 13 < u.elo_a_after - 1100 < 16


@pytest.mark.asyncio
async def test_apply_elo_update_is_idempotent(conn) -> None:
    """Re-applying the same match_id never double-counts."""
    now = datetime.now(UTC)

    # Set up a session row + two hypotheses in tournament
    await conn.execute(
        """INSERT INTO sessions(id, created_at, updated_at, status, research_goal,
                                 research_plan, config_snapshot, budget_tokens, budget_usd)
           VALUES ('ses_t', ?, ?, 'running', 'test', '{}', '{}', 1000000, 10.0)""",
        (now.isoformat(), now.isoformat()),
    )
    await conn.commit()

    for hid in ("hyp_x", "hyp_y"):
        h = Hypothesis(
            id=hid, session_id="ses_t", created_at=now,
            created_by="generation", strategy="literature",
            title="t", summary="s", full_text="f",
            artifact_path=f"artifacts/ses_t/hypotheses/{hid}.json",
            elo=1200, matches_played=0, state="in_tournament",
        )
        await hyp_repo.insert(conn, h)

    mid = ids.match_id("hyp_x", "hyp_y", "round1")
    m = TournamentMatch(
        id=mid, session_id="ses_t", created_at=now,
        hyp_a="hyp_x", hyp_b="hyp_y", mode="pairwise", winner="a",
        elo_a_before=1200.0, elo_b_before=1200.0,
        elo_a_after=1216.0, elo_b_after=1184.0, rationale="test",
    )
    await tourney_repo.insert_match(conn, m)

    ok1 = await tourney_repo.apply_elo_update(
        conn, match_id=mid, hyp_a="hyp_x", hyp_b="hyp_y", winner="a",
        elo_a_before=1200.0, elo_b_before=1200.0,
        elo_a_after=1216.0, elo_b_after=1184.0,
    )
    assert ok1 is True

    # Re-apply: should no-op.
    ok2 = await tourney_repo.apply_elo_update(
        conn, match_id=mid, hyp_a="hyp_x", hyp_b="hyp_y", winner="a",
        elo_a_before=1200.0, elo_b_before=1200.0,
        elo_a_after=1216.0, elo_b_after=1184.0,
    )
    assert ok2 is False

    # State reflects exactly one update
    hx = await hyp_repo.fetch(conn, "hyp_x")
    hy = await hyp_repo.fetch(conn, "hyp_y")
    assert hx is not None and hy is not None
    assert hx.elo == 1216.0
    assert hy.elo == 1184.0
    assert hx.matches_played == 1
    assert hy.matches_played == 1

@pytest.mark.asyncio
async def test_concurrent_elo_updates_do_not_share_nested_transactions(conn) -> None:
    now = datetime.now(UTC)
    await conn.execute(
        """INSERT INTO sessions(id, created_at, updated_at, status, research_goal,
                                 research_plan, config_snapshot, budget_tokens, budget_usd)
           VALUES ('ses_concurrent', ?, ?, 'running', 'test', '{}', '{}', 1000000, 10.0)""",
        (now.isoformat(), now.isoformat()),
    )
    await conn.commit()

    for hid in ("hyp_a", "hyp_b", "hyp_c", "hyp_d"):
        await hyp_repo.insert(conn, Hypothesis(
            id=hid,
            session_id="ses_concurrent",
            created_at=now,
            created_by="generation",
            strategy="literature",
            title="t",
            summary="s",
            full_text="f",
            artifact_path=f"artifacts/ses_concurrent/hypotheses/{hid}.json",
            elo=1200.0,
            matches_played=0,
            state="in_tournament",
        ))

    pairs = [("hyp_a", "hyp_b", "round1"), ("hyp_c", "hyp_d", "round1")]
    for hyp_a, hyp_b, round_id in pairs:
        mid = ids.match_id(hyp_a, hyp_b, round_id)
        await tourney_repo.insert_match(conn, TournamentMatch(
            id=mid,
            session_id="ses_concurrent",
            created_at=now,
            hyp_a=hyp_a,
            hyp_b=hyp_b,
            mode="pairwise",
            winner="a",
            elo_a_before=1200.0,
            elo_b_before=1200.0,
            elo_a_after=1216.0,
            elo_b_after=1184.0,
            rationale="test",
        ))

    results = await asyncio.gather(*(
        tourney_repo.apply_elo_update(
            conn,
            match_id=ids.match_id(hyp_a, hyp_b, round_id),
            hyp_a=hyp_a,
            hyp_b=hyp_b,
            winner="a",
            elo_a_before=1200.0,
            elo_b_before=1200.0,
            elo_a_after=1216.0,
            elo_b_after=1184.0,
        )
        for hyp_a, hyp_b, round_id in pairs
    ))

    assert results == [True, True]
    for hid, expected_elo in (
        ("hyp_a", 1216.0),
        ("hyp_b", 1184.0),
        ("hyp_c", 1216.0),
        ("hyp_d", 1184.0),
    ):
        h = await hyp_repo.fetch(conn, hid)
        assert h is not None
        assert h.elo == expected_elo
        assert h.matches_played == 1



@pytest.mark.asyncio
async def test_apply_elo_update_recomputes_from_current_db_ratings(conn) -> None:
    """Stale prompt-time Elo values must not overwrite intervening updates."""
    now = datetime.now(UTC)
    await conn.execute(
        """INSERT INTO sessions(id, created_at, updated_at, status, research_goal,
                                 research_plan, config_snapshot, budget_tokens, budget_usd)
           VALUES ('ses_stale', ?, ?, 'running', 'test', '{}', '{}', 1000000, 10.0)""",
        (now.isoformat(), now.isoformat()),
    )
    await conn.commit()

    for hid in ("hyp_x", "hyp_y"):
        await hyp_repo.insert(conn, Hypothesis(
            id=hid,
            session_id="ses_stale",
            created_at=now,
            created_by="generation",
            strategy="literature",
            title="t",
            summary="s",
            full_text="f",
            artifact_path=f"artifacts/ses_stale/hypotheses/{hid}.json",
            elo=1200.0,
            matches_played=0,
            state="in_tournament",
        ))

    mid1 = ids.match_id("hyp_x", "hyp_y", "round1")
    await tourney_repo.insert_match(conn, TournamentMatch(
        id=mid1,
        session_id="ses_stale",
        created_at=now,
        hyp_a="hyp_x",
        hyp_b="hyp_y",
        mode="pairwise",
        winner="a",
        elo_a_before=1200.0,
        elo_b_before=1200.0,
        elo_a_after=1216.0,
        elo_b_after=1184.0,
        rationale="first",
    ))
    assert await tourney_repo.apply_elo_update(
        conn,
        match_id=mid1,
        hyp_a="hyp_x",
        hyp_b="hyp_y",
        winner="a",
        elo_a_before=1200.0,
        elo_b_before=1200.0,
        elo_a_after=1216.0,
        elo_b_after=1184.0,
    ) is True

    stale_update = update_elo(1200.0, 1200.0, "a", matches_played_min=0)
    current_update = update_elo(1216.0, 1184.0, "a", matches_played_min=1)
    mid2 = ids.match_id("hyp_x", "hyp_y", "round2")
    await tourney_repo.insert_match(conn, TournamentMatch(
        id=mid2,
        session_id="ses_stale",
        created_at=now,
        hyp_a="hyp_x",
        hyp_b="hyp_y",
        mode="pairwise",
        winner="a",
        elo_a_before=1200.0,
        elo_b_before=1200.0,
        elo_a_after=stale_update.elo_a_after,
        elo_b_after=stale_update.elo_b_after,
        rationale="second",
    ))

    assert await tourney_repo.apply_elo_update(
        conn,
        match_id=mid2,
        hyp_a="hyp_x",
        hyp_b="hyp_y",
        winner="a",
        elo_a_before=1200.0,
        elo_b_before=1200.0,
        elo_a_after=stale_update.elo_a_after,
        elo_b_after=stale_update.elo_b_after,
    ) is True

    hx = await hyp_repo.fetch(conn, "hyp_x")
    hy = await hyp_repo.fetch(conn, "hyp_y")
    assert hx is not None and hy is not None
    assert hx.elo == pytest.approx(current_update.elo_a_after)
    assert hy.elo == pytest.approx(current_update.elo_b_after)
    assert hx.matches_played == 2
    assert hy.matches_played == 2

    async with conn.execute(
        """SELECT elo_a_before, elo_b_before, elo_a_after, elo_b_after
             FROM tournament_matches WHERE id=?""",
        (mid2,),
    ) as cur:
        row = await cur.fetchone()
    assert row is not None
    assert row["elo_a_before"] == pytest.approx(current_update.elo_a_before)
    assert row["elo_b_before"] == pytest.approx(current_update.elo_b_before)
    assert row["elo_a_after"] == pytest.approx(current_update.elo_a_after)
    assert row["elo_b_after"] == pytest.approx(current_update.elo_b_after)
