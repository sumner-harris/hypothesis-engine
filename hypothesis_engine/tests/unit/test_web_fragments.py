"""Tests for session-detail HTML fragments used by live UI refresh."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from hypothesis_engine.models import (
    CitedPaper,
    Hypothesis,
    ResearchPlan,
    Session,
    SystemFeedback,
    Task,
    TournamentMatch,
)
from hypothesis_engine.storage.repos import feedback as fb_repo
from hypothesis_engine.storage.repos import hypotheses as hyp_repo
from hypothesis_engine.storage.repos import sessions as sess_repo
from hypothesis_engine.storage.repos import tasks as task_repo
from hypothesis_engine.storage.repos import tournaments as tourney_repo
from hypothesis_engine.vectors.store import FaissStore
from hypothesis_engine.web.app import (
    _citation_text_stats,
    _recover_orphaned_running_sessions,
    create_app,
)


def _session() -> Session:
    now = datetime.now(UTC)
    return Session(
        id="ses_web",
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


def _hypothesis(
    hid: str,
    session_id: str,
    *,
    title: str,
    elo: float,
    citations: list[CitedPaper] | None = None,
    full_text: str = "full text",
) -> Hypothesis:
    return Hypothesis(
        id=hid,
        session_id=session_id,
        created_at=datetime.now(UTC),
        created_by="generation",
        strategy="literature",
        title=title,
        summary="summary",
        full_text=full_text,
        citations=citations or [],
        artifact_path=f"artifacts/{hid}.md",
        elo=elo,
        matches_played=1,
        state="in_tournament",
    )


async def test_session_fragments_render_leaderboard_and_matches(conn, tmp_cfg) -> None:
    session = _session()
    session.research_goal = "Full research goal " + ("mechanism context " * 20) + "goal-tail-marker"
    await sess_repo.insert(conn, session)
    await fb_repo.insert(
        conn,
        SystemFeedback(
            id="fb_web_prompt",
            session_id=session.id,
            created_at=datetime.now(UTC),
            source="human",
            kind="preference",
            text="Full additional preferences " + ("evidence standard " * 20) + "preference-tail-marker",
        ),
    )
    await hyp_repo.insert(conn, _hypothesis("hyp_a", session.id, title="Alpha", elo=1210.0))
    await hyp_repo.insert(conn, _hypothesis("hyp_b", session.id, title="Beta", elo=1190.0))
    vectors = FaissStore(tmp_cfg, session.id, dim=4)
    await vectors.add_and_save("hyp_a", np.asarray([1.0, 0.0, 0.0, 0.0], dtype="float32"))
    await vectors.add_and_save("hyp_b", np.asarray([0.0, 1.0, 0.0, 0.0], dtype="float32"))
    await tourney_repo.insert_match(
        conn,
        TournamentMatch(
            id="mat_web",
            session_id=session.id,
            created_at=datetime.now(UTC),
            hyp_a="hyp_a",
            hyp_b="hyp_b",
            mode="pairwise",
            winner="a",
            elo_a_before=1200.0,
            elo_b_before=1200.0,
            elo_a_after=1210.0,
            elo_b_after=1190.0,
        ),
    )
    await task_repo.enqueue(
        conn,
        Task(
            id="tsk_web_reflection",
            session_id=session.id,
            created_at=datetime.now(UTC),
            started_at=datetime.now(UTC),
            agent="reflection",
            action="ReviewHypothesis",
            target_id="hyp_a",
            payload={},
            status="in_progress",
        ),
    )
    papers_dir = tmp_cfg.session_artifact_dir(session.id) / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / "paper_a.json").write_text(
        json.dumps({"url": "https://example.test/a", "text": "alpha text"}),
        encoding="utf-8",
    )
    (papers_dir / "paper_b.json").write_text(
        json.dumps({"url": "https://example.test/b", "text": "beta text"}),
        encoding="utf-8",
    )
    (papers_dir / "paper_a_duplicate.json").write_text(
        json.dumps({"url": "https://example.test/a", "text": "duplicate text"}),
        encoding="utf-8",
    )
    (papers_dir / "empty.json").write_text(
        json.dumps({"url": "https://example.test/empty", "text": ""}),
        encoding="utf-8",
    )
    arxiv_dir = tmp_cfg.session_artifact_dir(session.id) / "searches" / "arxiv_search"
    arxiv_dir.mkdir(parents=True, exist_ok=True)
    (arxiv_dir / "search_a.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "pdf_url": "https://arxiv.org/pdf/1234.5678v1",
                        "title": "Gamma",
                        "summary": "gamma summary",
                    },
                    {
                        "pdf_url": "https://arxiv.org/pdf/1234.5678v1",
                        "title": "Gamma duplicate",
                        "summary": "duplicate gamma summary",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    europe_dir = tmp_cfg.session_artifact_dir(session.id) / "searches" / "europe_pmc_search"
    europe_dir.mkdir(parents=True, exist_ok=True)
    (europe_dir / "search_b.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "url": "https://europepmc.org/article/test",
                        "title": "Delta",
                        "abstract": "delta abstract",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rag_dir = tmp_cfg.session_rag_dir(session.id)
    pdf_dir = rag_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ("arxiv.pdf", "biorxiv.pdf", "chemrxiv.pdf", "europe.pdf", "other.pdf"):
        (pdf_dir / file_name).write_bytes(b"%PDF-1.4\n")
    (rag_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seed": {"kb": {"status": "ok", "chunk_count": 1241}},
                "kb": {"status": "ok", "chunk_count": 1300},
                "papers": {
                    "arxiv": {
                        "url": "https://arxiv.org/pdf/1234.5678v1",
                        "file": "arxiv.pdf",
                    },
                    "biorxiv": {
                        "url": "https://www.biorxiv.org/content/10.64898/2026.05.06.723241.full.pdf",
                        "file": "biorxiv.pdf",
                    },
                    "chemrxiv": {
                        "url": "https://www.cambridge.org/engage/api-gateway/coe/assets/orp/resource/item/x/original/paper.pdf",
                        "file": "chemrxiv.pdf",
                    },
                    "europe": {
                        "url": "https://europepmc.org/articles/PMC1234567?pdf=render",
                        "file": "europe.pdf",
                    },
                    "other": {
                        "url": "https://example.test/paper.pdf",
                        "file": "other.pdf",
                    },
                    "missing": {
                        "url": "https://arxiv.org/pdf/9999.9999v1",
                        "file": "missing.pdf",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    stats = await _citation_text_stats(tmp_cfg, session.id)
    assert stats == {
        "unique_texts": 4,
        "total_chars": 46,
        "pdfs_total": 5,
        "pdfs_arxiv": 1,
        "pdfs_biorxiv": 1,
        "pdfs_chemrxiv": 1,
        "pdfs_europe_pmc": 1,
        "pdfs_other": 1,
        "kb_chunks": 1300,
        "seed_kb_chunks": 1241,
        "kb_seeded": 1,
    }

    client = TestClient(create_app(tmp_cfg))

    status = client.get(f"/sessions/{session.id}/status")
    assert status.status_code == 200
    assert 'id="session-status-panel"' in status.text
    assert 'hx-get="/sessions/ses_web/status"' in status.text
    assert 'hx-trigger="every 5s"' in status.text
    assert "API calls" in status.text
    assert "Citation texts" in status.text
    assert "4 unique citation/search sources" in status.text
    assert "RAG KB: 1300 text chunks" in status.text
    assert "seed loaded (1241)" in status.text
    assert "PDFs: 5 total" in status.text
    assert "arXiv 1" in status.text
    assert "bioRxiv 1" in status.text
    assert "ChemRxiv 1" in status.text
    assert "Europe PMC 1" in status.text
    assert "other 1" in status.text

    leaderboard = client.get(f"/sessions/{session.id}/leaderboard")
    assert leaderboard.status_code == 200
    assert 'id="leaderboard-panel"' in leaderboard.text
    assert 'hx-get="/sessions/ses_web/leaderboard"' in leaderboard.text
    assert 'hx-trigger="every 5s"' in leaderboard.text
    assert "Alpha" in leaderboard.text
    assert "1210" in leaderboard.text

    matches = client.get(f"/sessions/{session.id}/matches")
    assert matches.status_code == 200
    assert 'id="matches-panel"' in matches.text
    assert 'hx-get="/sessions/ses_web/matches"' in matches.text
    assert 'hx-trigger="every 5s"' in matches.text
    assert "hyp_a" in matches.text
    assert "hyp_b" in matches.text

    active_lit_dir = tmp_cfg.session_artifact_dir(session.id) / "literature_review" / "active_calls"
    active_lit_dir.mkdir(parents=True, exist_ok=True)
    (active_lit_dir / "active.json").write_text(
        json.dumps({"trigger_agent": "reflection"}), encoding="utf-8"
    )

    detail = client.get(f"/sessions/{session.id}")
    assert detail.status_code == 200
    assert 'id="session-status-panel"' in detail.text
    assert 'id="leaderboard-panel"' in detail.text
    assert 'id="matches-panel"' in detail.text
    assert 'id="workflow-diagram"' in detail.text
    assert "Session prompt" in detail.text
    assert "Research goal" in detail.text
    assert "goal-tail-marker" in detail.text
    assert "Additional preferences" in detail.text
    assert "preference-tail-marker" in detail.text
    assert "data-session-tabs" in detail.text
    assert "Hypothesis clusters" in detail.text
    assert "RAG KB clusters" in detail.text
    assert f"/api/sessions/{session.id}/clusters?view=hypotheses" in detail.text
    assert f"/api/sessions/{session.id}/clusters?view=rag" in detail.text
    assert "4 unique citation/search sources" in detail.text
    assert "RAG KB: 1300 text chunks" in detail.text
    assert "seed loaded (1241)" in detail.text
    assert "PDFs: 5 total" in detail.text
    assert "bioRxiv 1" in detail.text
    assert 'data-workflow-stage="reflection"' in detail.text
    assert 'data-workflow-stage="literature_review"' in detail.text
    assert 'data-workflow-count-for="literature_review"' in detail.text
    assert "ReviewHypothesis" in detail.text
    assert "/api/sessions/ses_web/workflow" in detail.text
    assert "/leaderboard" in detail.text

    workflow = client.get(f"/api/sessions/{session.id}/workflow")
    assert workflow.status_code == 200
    assert workflow.json()["counts"]["reflection"] == 1
    assert workflow.json()["counts"]["literature_review"] == 1
    assert workflow.json()["active_literature_review_sources"] == ["reflection"]

    clusters = client.get(f"/api/sessions/{session.id}/clusters?view=hypotheses")
    assert clusters.status_code == 200
    cluster_body = clusters.json()
    assert cluster_body["available"] is True
    assert cluster_body["view"] == "hypotheses"
    assert cluster_body["metrics"]["points"] == 2
    assert cluster_body["metrics"]["cluster_count"] == 2
    assert {point["id"] for point in cluster_body["points"]} == {"hyp_a", "hyp_b"}
    assert any(point["label"] for point in cluster_body["points"])

    rag_clusters = client.get(f"/api/sessions/{session.id}/clusters?view=rag")
    assert rag_clusters.status_code == 200
    assert rag_clusters.json()["available"] is False


async def test_startup_recovery_pauses_orphaned_running_session(conn, tmp_cfg) -> None:
    session = _session()
    session.id = "ses_orphan"
    stale_time = datetime.now(UTC) - timedelta(minutes=30)
    session.created_at = stale_time
    session.updated_at = stale_time
    await sess_repo.insert(conn, session)
    await task_repo.enqueue(
        conn,
        Task(
            id="tsk_orphan",
            session_id=session.id,
            created_at=stale_time,
            started_at=stale_time,
            agent="reflection",
            action="ReviewHypothesis",
            target_id="hyp_missing",
            payload={},
            status="pending",
        ),
    )
    await conn.execute(
        """UPDATE tasks
              SET status='in_progress', lease_owner='dead-worker', lease_expires_at=?
            WHERE id='tsk_orphan'""",
        (int(time.time() * 1000) - 60_000,),
    )
    await conn.commit()

    summary = await _recover_orphaned_running_sessions(tmp_cfg)

    assert summary == {"paused_sessions": 1, "requeued_tasks": 1, "dead_tasks": 0}
    refreshed = await sess_repo.fetch(conn, session.id)
    assert refreshed is not None
    assert refreshed.status == "paused"
    async with conn.execute(
        "SELECT status, attempts, lease_owner, lease_expires_at, started_at FROM tasks WHERE id='tsk_orphan'"
    ) as cur:
        row = await cur.fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None
    assert row["started_at"] is None


async def test_startup_recovery_preserves_running_session_with_live_lease(conn, tmp_cfg) -> None:
    session = _session()
    session.id = "ses_live"
    stale_time = datetime.now(UTC) - timedelta(minutes=30)
    session.created_at = stale_time
    session.updated_at = stale_time
    await sess_repo.insert(conn, session)
    await task_repo.enqueue(
        conn,
        Task(
            id="tsk_live",
            session_id=session.id,
            created_at=stale_time,
            started_at=stale_time,
            agent="reflection",
            action="ReviewHypothesis",
            target_id="hyp_live",
            payload={},
            status="pending",
        ),
    )
    await conn.execute(
        """UPDATE tasks
              SET status='in_progress', lease_owner='worker', lease_expires_at=?
            WHERE id='tsk_live'""",
        (int(time.time() * 1000) + 60_000,),
    )
    await conn.commit()

    summary = await _recover_orphaned_running_sessions(tmp_cfg)

    assert summary == {"paused_sessions": 0, "requeued_tasks": 0, "dead_tasks": 0}
    refreshed = await sess_repo.fetch(conn, session.id)
    assert refreshed is not None
    assert refreshed.status == "running"
    async with conn.execute("SELECT status, attempts FROM tasks WHERE id='tsk_live'") as cur:
        row = await cur.fetchone()
    assert row["status"] == "in_progress"
    assert row["attempts"] == 0


async def test_hypothesis_detail_shows_stored_citations(conn, tmp_cfg) -> None:
    session = _session()
    session.id = "ses_web_citations"
    await sess_repo.insert(conn, session)
    await hyp_repo.insert(
        conn,
        _hypothesis(
            "hyp_cited",
            session.id,
            title="Cited hypothesis",
            elo=1200.0,
            citations=[
                CitedPaper(
                    title="Displayed source",
                    url="https://example.test/displayed",
                    excerpt="Relevant stored citation excerpt.",
                    year=2026,
                )
            ],
            full_text=(
                "Main hypothesis body.\n\n"
                "## Citations\n\n"
                "- Generated duplicate citation - https://example.test/generated"
            ),
        ),
    )

    client = TestClient(create_app(tmp_cfg))
    response = client.get(f"/sessions/{session.id}/hypotheses/hyp_cited")

    assert response.status_code == 200
    assert "Stored citation sources" not in response.text
    assert response.text.count("<h2>Citations</h2>") == 1
    assert response.text.index("Main hypothesis body.") < response.text.index("<h2>Citations</h2>")
    assert response.text.index("<h2>Citations</h2>") < response.text.index("<h2>Reviews")
    assert "Generated duplicate citation" not in response.text
    assert "Displayed source" in response.text
    assert "https://example.test/displayed" in response.text
    assert "Relevant stored citation excerpt." in response.text


async def test_final_overview_page_loads_math_renderer(conn, tmp_cfg) -> None:
    session = _session()
    session.id = "ses_overview_math"
    overview_rel = "artifacts/ses_overview_math/final/overview.md"
    session.final_overview = overview_rel
    await sess_repo.insert(conn, session)

    overview_path = tmp_cfg.data_dir / overview_rel
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview_path.write_text("# Overview\n\nImplantation uses \\text{Ar}^+ ions.", encoding="utf-8")

    client = TestClient(create_app(tmp_cfg))
    response = client.get(f"/sessions/{session.id}/overview")

    assert response.status_code == 200
    assert 'class="overview-content"' in response.text
    assert "data-render-math" in response.text
    assert "/static/overview_math.js" in response.text
    assert "mathjax@3/es5/tex-chtml.js" in response.text
    assert "\\text{Ar}^+" in response.text

    asset = client.get("/static/overview_math.js")
    assert asset.status_code == 200
    assert "wrapBareTex" in asset.text
