# Modified from the original work.
"""FastAPI web UI for the hypothesis-engine.

One process per host: launching `hypothesis-engine serve` runs both the API + UI
and the worker pool — the queue is DB-backed so CLI `hypothesis-engine run` in a
separate terminal feeds tasks to whatever Supervisor is currently active.

The UI is server-side Jinja2 + htmx for partial updates + SSE for live events.
No JS build step. Pico.css for default styling.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging as stdlib_logging
import pickle
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from .. import ids
from ..citations import render_citations_md
from ..config import Config, load_config
from ..logging import get_logger
from ..models import SystemFeedback
from ..orchestrator.events import GLOBAL_BUS
from ..orchestrator.feedback_actions import apply_human_feedback_actions
from ..storage import db as db_mod
from ..storage.repos import events as events_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import transcripts as tx_repo
from .sanitize import render_markdown
from .workflow_state import WORKFLOW_STAGES, workflow_state

log = get_logger("web")
HERE = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=HERE / "templates")
CLUSTER_PLOT_MAX_POINTS = 1_600
CLUSTER_PLOT_MAX_CLUSTERS = 12
CLUSTER_PLOT_TOP_LABELS = 10
_TRAILING_CITATIONS_HEADING_RE = re.compile(r"(?im)^#{1,6}\s+Citations\s*:?\s*$")


def _strip_trailing_citations_section(markdown: str) -> str:
    matches = list(_TRAILING_CITATIONS_HEADING_RE.finditer(markdown or ""))
    if not matches:
        return markdown or ""
    return (markdown or "")[: matches[-1].start()].rstrip()


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await db_mod.init_db(cfg)
        summary = await _recover_orphaned_running_sessions(cfg)
        if summary["paused_sessions"]:
            log.warning("orphaned_running_sessions_paused", **summary)
        yield

    app = FastAPI(title="Hypothesis Engine", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.background_runs: dict[str, asyncio.Task] = {}

    # Static
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    # ----------------------------- pages ----------------------------- #

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        rows = await _list_sessions(cfg)
        return TEMPLATES.TemplateResponse(request, "index.html", {"sessions": rows})

    @app.get("/dashboard/sessions", response_class=HTMLResponse)
    async def dashboard_sessions(request: Request) -> HTMLResponse:
        rows = await _list_sessions(cfg)
        return TEMPLATES.TemplateResponse(request, "_sessions_table.html", {"sessions": rows})

    @app.get("/sessions/new", response_class=HTMLResponse)
    async def new_session_form(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "new_session.html",
            {
                "default_budget": cfg.run.budget_usd,
                "default_n_initial": cfg.run.initial_generations,
                "max_n_initial": cfg.run.max_ideas,
                "default_wall_clock_seconds": cfg.run.wall_clock_seconds,
            },
        )

    @app.post("/sessions/new")
    async def create_session(
        request: Request,
        background_tasks: BackgroundTasks,
        goal: str = Form(...),
        preferences: str = Form(""),
        budget_usd: float = Form(cfg.run.budget_usd),
        n_initial: int = Form(cfg.run.initial_generations),
        wall_clock_seconds: int = Form(cfg.run.wall_clock_seconds),
    ) -> RedirectResponse:
        from ..agents.supervisor import Supervisor

        # Hand the Supervisor a fresh Config copy so per-session knobs don't leak.
        sup_cfg = cfg.model_copy(deep=True)
        sup_cfg.run.budget_usd = budget_usd
        sup_cfg.run.wall_clock_seconds = wall_clock_seconds
        sup = Supervisor(sup_cfg)

        async def _run() -> None:
            try:
                await sup.run_session(
                    goal=goal,
                    preferences_text=preferences or None,
                    n_initial=n_initial,
                    wall_clock_seconds=wall_clock_seconds,
                )
            except Exception:
                log.exception("background_run_failed")

        task = asyncio.create_task(_run())
        # No durable session id at this point — give the run a chance to insert
        # the row, then redirect to /. The user can find it in the listing.
        _ = task
        return RedirectResponse(url="/", status_code=303)

    @app.get("/sessions/{session_id}", response_class=HTMLResponse)
    async def session_detail(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            hyps = await hyp_repo.list_for_session(conn, session_id)
            recent_matches = await _recent_matches(conn, session_id, limit=20)
            usage = await tx_repo.usage_summary(conn, session_id)
            citation_text_stats = await _citation_text_stats(cfg, session_id)
            prompt_preferences = await fb_repo.human_preferences_for_session(conn, session_id)
            wf_state = await workflow_state(conn, session_id, cfg)
            return TEMPLATES.TemplateResponse(
                request,
                "session_detail.html",
                {
                    "session": session,
                    "hypotheses": sorted(hyps, key=lambda h: -(h.elo or 0)),
                    "recent_matches": recent_matches,
                    "usage": usage,
                    "citation_text_stats": citation_text_stats,
                    "prompt_preferences": prompt_preferences,
                    "workflow_stages": WORKFLOW_STAGES,
                    "workflow_state": wf_state,
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/status", response_class=HTMLResponse)
    async def session_status(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            usage = await tx_repo.usage_summary(conn, session_id)
            citation_text_stats = await _citation_text_stats(cfg, session_id)
            return TEMPLATES.TemplateResponse(
                request,
                "_session_status.html",
                {
                    "session": session,
                    "usage": usage,
                    "citation_text_stats": citation_text_stats,
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/leaderboard", response_class=HTMLResponse)
    async def session_leaderboard(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            hyps = await hyp_repo.list_for_session(conn, session_id)
            return TEMPLATES.TemplateResponse(
                request,
                "_leaderboard.html",
                {
                    "session": session,
                    "hypotheses": sorted(hyps, key=lambda h: -(h.elo or 0)),
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/matches", response_class=HTMLResponse)
    async def session_matches(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            recent_matches = await _recent_matches(conn, session_id, limit=20)
            return TEMPLATES.TemplateResponse(
                request,
                "_matches.html",
                {"session": session, "recent_matches": recent_matches},
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/hypotheses/{hid}", response_class=HTMLResponse)
    async def hypothesis_detail(request: Request, session_id: str, hid: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            h = await hyp_repo.fetch(conn, hid)
            session = await sess_repo.fetch(conn, session_id)
            if h is None or session is None:
                raise HTTPException(status_code=404, detail="not found")
            reviews = await rev_repo.list_for_hypothesis(conn, hid)
            citations_md = render_citations_md(h.citations, heading="## Citations")
            full_text_md = h.full_text or ""
            if h.citations:
                full_text_md = _strip_trailing_citations_section(full_text_md)
            return TEMPLATES.TemplateResponse(
                request,
                "hypothesis_detail.html",
                {
                    "session": session,
                    "h": h,
                    "reviews": reviews,
                    "citations_html": render_markdown(citations_md),
                    "full_text_html": render_markdown(full_text_md),
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/overview", response_class=HTMLResponse)
    async def session_overview(request: Request, session_id: str) -> HTMLResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None or not session.final_overview:
                raise HTTPException(
                    status_code=404, detail="no final overview yet for this session"
                )
            # `final_overview` is written by the supervisor under
            # `data_dir/artifacts/...` but is stored as a string in the DB.
            # Resolve and confirm the path is still inside `data_dir` so a
            # tampered row can't read arbitrary files.
            base = cfg.data_dir.resolve()
            try:
                path = (cfg.data_dir / session.final_overview).resolve()
                path.relative_to(base)
            except (ValueError, OSError) as e:
                log.error("overview_path_escape", session=session_id, err=str(e))
                raise HTTPException(status_code=404, detail="overview unavailable") from e
            if not path.is_file():
                raise HTTPException(status_code=404, detail="overview missing on disk")
            overview_md = path.read_text(encoding="utf-8")
            return TEMPLATES.TemplateResponse(
                request,
                "overview.html",
                {
                    "session": session,
                    "overview_html": render_markdown(overview_md),
                    "overview_md": overview_md,
                },
            )
        finally:
            await conn.close()

    @app.get("/sessions/{session_id}/analysis", response_class=HTMLResponse)
    async def session_analysis(request: Request, session_id: str) -> HTMLResponse:
        await _ensure_session_exists(cfg, session_id)
        report_path = await _ensure_analysis_report(cfg, session_id)
        html_path = report_path.parent / "report.html"
        if not html_path.is_file():
            raise HTTPException(status_code=404, detail="analysis report unavailable")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/sessions/{session_id}/analysis/download")
    async def session_analysis_download(session_id: str) -> FileResponse:
        await _ensure_session_exists(cfg, session_id)
        report_path = await _ensure_analysis_report(cfg, session_id)
        bundle_path = report_path.parent / "analysis_report.zip"
        if not bundle_path.is_file():
            raise HTTPException(status_code=404, detail="analysis bundle unavailable")
        return FileResponse(
            bundle_path,
            media_type="application/zip",
            filename=f"hypothesis_engine_analysis_{session_id}.zip",
        )

    # ----------------------------- API + SSE ----------------------------- #

    @app.get("/api/sessions/{session_id}/metrics")
    async def api_metrics(session_id: str) -> JSONResponse:
        from ..obs.metrics import session_metrics_cached, to_dict

        conn = await db_mod.connect(cfg)
        try:
            m = await session_metrics_cached(conn, session_id)
            return JSONResponse(to_dict(m))
        finally:
            await conn.close()

    @app.get("/api/sessions/{session_id}")
    async def api_session(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            s = await sess_repo.fetch(conn, session_id)
            if s is None:
                raise HTTPException(status_code=404)
            return JSONResponse(s.model_dump(mode="json"))
        finally:
            await conn.close()

    @app.get("/api/sessions/{session_id}/events")
    async def api_events(session_id: str) -> EventSourceResponse:
        async def _stream() -> AsyncIterator[dict[str, Any]]:
            # Replay recent history, then poll the durable event log. The
            # in-memory event bus is process-local, but users commonly run the
            # supervisor from the CLI while viewing this separate web process.
            conn = await db_mod.connect(cfg)
            last_id = 0
            try:
                history = await events_repo.recent(conn, session_id, limit=25)
                ordered_history = list(reversed(history))
                for ev in ordered_history:
                    last_id = max(last_id, int(ev["id"]))
                    yield _sse_event(ev)

                while True:
                    rows = await events_repo.after_id(conn, session_id, last_id, limit=100)
                    if not rows:
                        await asyncio.sleep(1.0)
                        continue
                    for ev in rows:
                        last_id = max(last_id, int(ev["id"]))
                        yield _sse_event(ev)
            finally:
                await conn.close()

        return EventSourceResponse(_stream())

    @app.get("/api/sessions/{session_id}/workflow")
    async def api_workflow(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            return JSONResponse(await workflow_state(conn, session_id, cfg))
        finally:
            await conn.close()

    @app.get("/api/sessions/{session_id}/clusters")
    async def api_clusters(session_id: str, view: str = "hypotheses") -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
            hyps = await hyp_repo.list_for_session(conn, session_id)
        finally:
            await conn.close()

        payload = await asyncio.to_thread(
            _cluster_plot_payload_sync,
            cfg,
            session_id,
            view,
            hyps,
        )
        return JSONResponse(payload)

    @app.post("/api/sessions/{session_id}/pause")
    async def api_pause(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            await sess_repo.set_status(conn, session_id, "paused")
            await _persist_and_publish(conn, session_id, "session_paused", {})
            return JSONResponse({"ok": True})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/resume")
    async def api_resume(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            session = await sess_repo.fetch(conn, session_id)
            if session is None:
                raise HTTPException(status_code=404)
            await sess_repo.set_status(conn, session_id, "running")
            await _persist_and_publish(conn, session_id, "session_resumed", {})
            already_running = _background_run_active(app, session_id)
            if not already_running:
                _start_background_resume(app, cfg, session_id)
            return JSONResponse({"ok": True, "already_running": already_running})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/abort")
    async def api_abort(session_id: str) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            await sess_repo.set_status(conn, session_id, "aborted")
            await _persist_and_publish(conn, session_id, "session_aborted", {})
            return JSONResponse({"ok": True})
        finally:
            await conn.close()

    @app.post("/api/sessions/{session_id}/feedback")
    async def api_feedback(
        session_id: str,
        text: str = Form(...),
        kind: str = Form("directive"),
        target_id: str = Form(""),
    ) -> JSONResponse:
        conn = await db_mod.connect(cfg)
        try:
            fb = SystemFeedback(
                id=ids.feedback_id(),
                session_id=session_id,
                created_at=datetime.now(UTC),
                source="human",
                kind=kind,
                target_id=target_id or None,
                text=text,
                active=True,
            )
            await fb_repo.insert(conn, fb)
            actions = await apply_human_feedback_actions(
                conn,
                session_id=session_id,
                feedback_id=fb.id,
                kind=kind,
                target_id=target_id or None,
            )
            await _persist_and_publish(
                conn,
                session_id,
                "human_feedback",
                {
                    "kind": kind,
                    "target_id": target_id or None,
                    "text": text[:200],
                    "actions": actions,
                },
            )
            return JSONResponse({"ok": True, "feedback_id": fb.id, "actions": actions})
        finally:
            await conn.close()

    @app.get("/healthz")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    # quiet uvicorn access spam during streaming
    stdlib_logging.getLogger("uvicorn.access").setLevel(stdlib_logging.WARNING)
    return app


# ----------------------------- helpers ----------------------------- #


async def _ensure_session_exists(cfg: Config, session_id: str) -> None:
    conn = await db_mod.connect(cfg)
    try:
        session = await sess_repo.fetch(conn, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
    finally:
        await conn.close()


async def _ensure_analysis_report(cfg: Config, session_id: str) -> Path:
    analysis_dir = cfg.session_artifact_dir(session_id) / "analysis"
    report_path = analysis_dir / "report.md"
    html_path = analysis_dir / "report.html"
    bundle_path = analysis_dir / "analysis_report.zip"
    if report_path.is_file() and html_path.is_file() and bundle_path.is_file():
        return report_path

    from ..analysis import analyze_session

    return await asyncio.to_thread(analyze_session, cfg, session_id)


def _background_run_active(app: FastAPI, session_id: str) -> bool:
    task = app.state.background_runs.get(session_id)
    if task is None:
        return False
    if task.done():
        app.state.background_runs.pop(session_id, None)
        return False
    return True


def _start_background_resume(app: FastAPI, cfg: Config, session_id: str) -> None:
    async def _run() -> None:
        from ..agents.supervisor import Supervisor

        try:
            await Supervisor(cfg.model_copy(deep=True)).run_session(
                goal="",
                resume_session_id=session_id,
            )
        except Exception:
            log.exception("background_resume_failed", session_id=session_id)
            conn = await db_mod.connect(cfg)
            try:
                await sess_repo.set_status(conn, session_id, "failed")
                await _persist_and_publish(
                    conn,
                    session_id,
                    "session_failed",
                    {"reason": "background_resume_failed"},
                )
            finally:
                await conn.close()
        finally:
            current = asyncio.current_task()
            if app.state.background_runs.get(session_id) is current:
                app.state.background_runs.pop(session_id, None)

    app.state.background_runs[session_id] = asyncio.create_task(_run())


def _cluster_plot_payload_sync(
    cfg: Config,
    session_id: str,
    view: str,
    hypotheses: list[Any],
) -> dict[str, Any]:
    view = (view or "hypotheses").strip().lower()
    if view not in {"hypotheses", "rag"}:
        return {
            "available": False,
            "view": view,
            "error": "unknown cluster view",
            "points": [],
            "overlays": [],
            "clusters": [],
        }
    try:
        if view == "hypotheses":
            return _hypothesis_cluster_plot_payload(cfg, session_id, hypotheses)
        return _rag_cluster_plot_payload(cfg, session_id, hypotheses)
    except Exception as exc:
        log.exception("cluster_plot_failed", session_id=session_id, view=view)
        return {
            "available": False,
            "view": view,
            "error": str(exc),
            "points": [],
            "overlays": [],
            "clusters": [],
        }


def _hypothesis_cluster_plot_payload(
    cfg: Config,
    session_id: str,
    hypotheses: list[Any],
) -> dict[str, Any]:
    loaded = _load_hypothesis_vectors_sync(cfg, session_id)
    if not loaded["available"]:
        return _cluster_unavailable(
            "hypotheses", loaded.get("error") or "missing hypothesis vectors"
        )

    by_id = {h.id: h for h in hypotheses}
    ids: list[str] = []
    vectors = []
    for hid, vec in zip(loaded["ids"], loaded["vectors"], strict=False):
        if hid in by_id:
            ids.append(hid)
            vectors.append(vec)
    if len(ids) < 2:
        return _cluster_unavailable("hypotheses", "need at least two hypothesis vectors")

    import numpy as np

    matrix = np.vstack(vectors).astype("float32")
    projection = _pca_kmeans_projection(matrix)
    embedded_ids = set(ids)
    ranked_ids = [
        h.id
        for h in sorted(hypotheses, key=lambda h: float(h.elo or 0.0), reverse=True)
        if h.id in embedded_ids
    ]
    rank_by_id = {hid: rank for rank, hid in enumerate(ranked_ids, 1)}
    labels_to_show = set(ranked_ids[:CLUSTER_PLOT_TOP_LABELS])
    points = []
    for idx, hid in enumerate(ids):
        h = by_id[hid]
        rank = rank_by_id.get(hid)
        points.append(
            {
                "id": hid,
                "kind": "hypothesis",
                "x": _finite_float(projection["plot"][idx, 0]),
                "y": _finite_float(projection["plot"][idx, 1]),
                "cluster": int(projection["labels"][idx]),
                "label": hid in labels_to_show,
                "rank": rank,
                "elo": _finite_float(h.elo),
                "matches_played": int(h.matches_played or 0),
                "title": h.title,
                "state": h.state,
                "created_by": h.created_by,
            }
        )

    return {
        "available": True,
        "view": "hypotheses",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "points": len(points),
            "vectors": loaded["count"],
            "cluster_count": projection["cluster_count"],
            "silhouette": _finite_float(projection["silhouette"]),
            "embedding_coverage": len(points) / len(hypotheses) if hypotheses else None,
            "top_label_count": len(labels_to_show),
        },
        "points": points,
        "overlays": [],
        "clusters": _cluster_rows(projection["labels"]),
    }


def _rag_cluster_plot_payload(
    cfg: Config,
    session_id: str,
    hypotheses: list[Any],
) -> dict[str, Any]:
    kb = _load_rag_sample_sync(cfg, session_id, max_points=CLUSTER_PLOT_MAX_POINTS)
    if not kb["available"]:
        return _cluster_unavailable("rag", kb.get("error") or "missing RAG KB vectors")
    if len(kb["meta"]) < 2:
        return _cluster_unavailable("rag", "need at least two RAG KB chunks")

    projection = _pca_kmeans_projection(kb["vectors"])
    points = []
    for coords, label, meta, sample_index in zip(
        projection["plot"], projection["labels"], kb["meta"], kb["sample_indices"], strict=False
    ):
        source = str(meta.get("source") or meta.get("file") or meta.get("url") or "")
        text = str(meta.get("text") or "")
        points.append(
            {
                "id": f"kb_{sample_index}",
                "kind": "kb_chunk",
                "x": _finite_float(coords[0]),
                "y": _finite_float(coords[1]),
                "cluster": int(label),
                "source": source,
                "chunk_id": meta.get("chunk_id"),
                "title": _short_source_name(source),
                "snippet": text[:220],
                "text_chars": len(text),
            }
        )

    overlays = _top_hypothesis_overlays(cfg, session_id, hypotheses, projection)
    high_counts: dict[int, int] = {}
    for item in overlays:
        cluster = item.get("cluster")
        if isinstance(cluster, int):
            high_counts[cluster] = high_counts.get(cluster, 0) + 1

    return {
        "available": True,
        "view": "rag",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": {
            "points": len(points),
            "kb_chunks": kb["count"],
            "sampled_chunks": len(points),
            "cluster_count": projection["cluster_count"],
            "silhouette": _finite_float(projection["silhouette"]),
            "top_label_count": len(overlays),
        },
        "points": points,
        "overlays": overlays,
        "clusters": _cluster_rows(projection["labels"], high_counts=high_counts),
    }


def _load_hypothesis_vectors_sync(cfg: Config, session_id: str) -> dict[str, Any]:
    try:
        import faiss
    except Exception as exc:
        return {"available": False, "error": str(exc), "ids": [], "vectors": [], "count": 0}

    vector_dir = cfg.session_vector_dir(session_id)
    index_path = vector_dir / "index.faiss"
    meta_path = vector_dir / "index.meta.json"
    if not index_path.exists() or not meta_path.exists():
        return {
            "available": False,
            "error": "missing FAISS index",
            "ids": [],
            "vectors": [],
            "count": 0,
        }
    try:
        idx = faiss.read_index(str(index_path))
        meta = _load_json_object(meta_path)
        ordered_ids = list(meta.get("ordered_ids", [])) if isinstance(meta, dict) else []
        n = min(int(idx.ntotal), len(ordered_ids))
        if n <= 0:
            return {
                "available": False,
                "error": "empty FAISS index",
                "ids": [],
                "vectors": [],
                "count": 0,
            }
        vectors = idx.reconstruct_n(0, n).astype("float32")
        return {
            "available": True,
            "ids": ordered_ids[:n],
            "vectors": _normalize_rows(vectors),
            "count": n,
            "dim": int(vectors.shape[1]),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "ids": [], "vectors": [], "count": 0}


def _load_rag_sample_sync(cfg: Config, session_id: str, *, max_points: int) -> dict[str, Any]:
    try:
        import faiss
        import numpy as np
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "vectors": [],
            "meta": [],
            "sample_indices": [],
        }

    rag_dir = cfg.session_rag_dir(session_id)
    index_path = rag_dir / "kb.index"
    meta_path = rag_dir / "kb.pkl"
    if not index_path.exists() or not meta_path.exists():
        return {
            "available": False,
            "error": "missing RAG KB index",
            "vectors": [],
            "meta": [],
            "sample_indices": [],
        }
    try:
        with meta_path.open("rb") as handle:
            metadata = pickle.load(handle)
        if not isinstance(metadata, list):
            metadata = []
        idx = faiss.read_index(str(index_path))
        n = min(int(idx.ntotal), len(metadata))
        if n <= 0:
            return {
                "available": False,
                "error": "empty RAG KB index",
                "vectors": [],
                "meta": [],
                "sample_indices": [],
            }
        sample_n = min(max(2, int(max_points)), n)
        rng = np.random.default_rng(13)
        sample_indices = (
            np.sort(rng.choice(n, size=sample_n, replace=False)) if sample_n < n else np.arange(n)
        )
        vectors = np.vstack([idx.reconstruct(int(i)) for i in sample_indices]).astype("float32")
        sample_meta = [
            metadata[int(i)] if isinstance(metadata[int(i)], dict) else {} for i in sample_indices
        ]
        return {
            "available": True,
            "vectors": _normalize_rows(vectors),
            "meta": sample_meta,
            "sample_indices": [int(i) for i in sample_indices],
            "count": n,
            "dim": int(vectors.shape[1]),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "vectors": [],
            "meta": [],
            "sample_indices": [],
        }


def _top_hypothesis_overlays(
    cfg: Config,
    session_id: str,
    hypotheses: list[Any],
    kb_projection: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    loaded = _load_hypothesis_vectors_sync(cfg, session_id)
    if not loaded["available"] or loaded.get("dim") != kb_projection.get("dim"):
        return []
    by_id = {h.id: h for h in hypotheses}
    vector_by_id = {
        hid: vec for hid, vec in zip(loaded["ids"], loaded["vectors"], strict=False) if hid in by_id
    }
    ranked = [
        h
        for h in sorted(hypotheses, key=lambda h: float(h.elo or 0.0), reverse=True)
        if h.id in vector_by_id
    ][:CLUSTER_PLOT_TOP_LABELS]
    if not ranked:
        return []

    vectors = np.vstack([vector_by_id[h.id] for h in ranked]).astype("float32")
    pca = kb_projection.get("pca")
    kmeans = kb_projection.get("kmeans")
    if pca is None:
        return []
    cluster_coords = pca.transform(vectors)
    plot = _pad_plot_coords(cluster_coords[:, :2])
    labels = kmeans.predict(cluster_coords) if kmeans is not None else [None] * len(ranked)
    overlays = []
    for rank, h in enumerate(ranked, 1):
        overlays.append(
            {
                "id": h.id,
                "kind": "hypothesis",
                "x": _finite_float(plot[rank - 1, 0]),
                "y": _finite_float(plot[rank - 1, 1]),
                "cluster": int(labels[rank - 1]) if labels[rank - 1] is not None else None,
                "rank": rank,
                "label": f"H{rank}",
                "elo": _finite_float(h.elo),
                "title": h.title,
                "matches_played": int(h.matches_played or 0),
            }
        )
    return overlays


def _pca_kmeans_projection(vectors: Any) -> dict[str, Any]:
    import warnings

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import silhouette_score

    vectors = _normalize_rows(np.asarray(vectors, dtype="float32"))
    n_samples, n_features = vectors.shape
    if n_samples <= 1:
        labels = np.zeros((n_samples,), dtype=int)
        return {
            "plot": np.zeros((n_samples, 2), dtype="float32"),
            "labels": labels,
            "cluster_count": len(set(labels.tolist())),
            "silhouette": None,
            "pca": None,
            "kmeans": None,
            "dim": n_features,
        }

    n_components = min(10, n_features, n_samples - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=13)
    cluster_coords = pca.fit_transform(vectors)
    plot = _pad_plot_coords(cluster_coords[:, :2])

    max_k = min(CLUSTER_PLOT_MAX_CLUSTERS, n_samples)
    best_labels: Any = np.zeros((n_samples,), dtype=int)
    best_model: Any = None
    best_score: float | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=13, n_init=10)
            labels = model.fit_predict(cluster_coords).astype(int)
            unique = set(labels.tolist())
            if len(unique) < 2:
                continue
            if len(unique) >= n_samples:
                score = -1.0
            else:
                score = float(silhouette_score(cluster_coords, labels))
            if best_score is None or score > best_score:
                best_score = score
                best_labels = labels
                best_model = model

    return {
        "plot": plot,
        "labels": best_labels,
        "cluster_count": len(set(best_labels.tolist())),
        "silhouette": best_score,
        "pca": pca,
        "kmeans": best_model,
        "dim": n_features,
    }


def _normalize_rows(vectors: Any) -> Any:
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2 or arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _pad_plot_coords(coords: Any) -> Any:
    import numpy as np

    arr = np.asarray(coords, dtype="float32")
    if arr.ndim != 2:
        return np.zeros((0, 2), dtype="float32")
    if arr.shape[1] >= 2:
        return arr[:, :2]
    return np.column_stack([arr[:, 0], np.zeros((arr.shape[0],), dtype="float32")])


def _cluster_rows(
    labels: Any, *, high_counts: dict[int, int] | None = None
) -> list[dict[str, Any]]:
    from collections import Counter

    high_counts = high_counts or {}
    counts = Counter(int(label) for label in labels)
    return [
        {
            "cluster": cluster,
            "size": counts[cluster],
            "high_performing_hypotheses": int(high_counts.get(cluster, 0)),
        }
        for cluster in sorted(counts)
    ]


def _cluster_unavailable(view: str, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "view": view,
        "error": reason,
        "points": [],
        "overlays": [],
        "clusters": [],
        "metrics": {},
    }


def _short_source_name(source: str) -> str:
    if not source:
        return "unknown"
    name = Path(source).name
    for suffix in (".pdf", ".json", ".txt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name[:80]


def _finite_float(value: Any) -> float | None:
    import math

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def _recover_orphaned_running_sessions(cfg: Config) -> dict[str, int]:
    """Pause runs that are marked running but have no live task lease.

    The web process does not own supervisors from a previous process. If a
    process was killed, its leased tasks can otherwise keep the workflow diagram
    and session list looking active forever.
    """
    now_ms = int(time.time() * 1000)
    stale_before = datetime.fromtimestamp(
        (now_ms - cfg.lease.default_seconds * 1000) / 1000,
        tz=UTC,
    )
    conn = await db_mod.connect(cfg)
    try:
        async with conn.execute(
            """SELECT s.id, s.updated_at
                 FROM sessions AS s
                WHERE s.status='running'
                  AND NOT EXISTS (
                      SELECT 1
                        FROM tasks AS t
                       WHERE t.session_id=s.id
                         AND t.status IN ('leased', 'in_progress')
                         AND t.lease_expires_at IS NOT NULL
                         AND t.lease_expires_at >= ?
                  )""",
            (now_ms,),
        ) as cur:
            rows = await cur.fetchall()

        stale_ids: list[str] = []
        for row in rows:
            try:
                updated_at = datetime.fromisoformat(row["updated_at"])
            except (TypeError, ValueError):
                updated_at = datetime.fromtimestamp(0, tz=UTC)
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            if updated_at <= stale_before:
                stale_ids.append(row["id"])

        if not stale_ids:
            return {"paused_sessions": 0, "requeued_tasks": 0, "dead_tasks": 0}

        task_counts = await _requeue_or_dead_orphaned_tasks(
            conn,
            stale_ids,
            max_attempts=cfg.lease.max_attempts,
        )

        placeholders = ",".join("?" for _ in stale_ids)
        await conn.execute(
            f"UPDATE sessions SET status='paused', updated_at=? WHERE id IN ({placeholders})",
            (datetime.now(UTC).isoformat(), *stale_ids),
        )
        await conn.commit()
        for session_id in stale_ids:
            await events_repo.emit(
                conn,
                session_id=session_id,
                task_id=None,
                agent="web",
                event="session_paused",
                payload={"reason": "startup_recovery"},
            )
        return {
            "paused_sessions": len(stale_ids),
            "requeued_tasks": task_counts["requeued"],
            "dead_tasks": task_counts["dead"],
        }
    finally:
        await conn.close()


async def _requeue_or_dead_orphaned_tasks(
    conn,
    session_ids: list[str],
    *,
    max_attempts: int,
) -> dict[str, int]:
    if not session_ids:
        return {"requeued": 0, "dead": 0}

    placeholders = ",".join("?" for _ in session_ids)
    active_statuses = ("leased", "in_progress")
    status_placeholders = ",".join("?" for _ in active_statuses)

    cur = await conn.execute(
        f"""UPDATE tasks
               SET status='dead',
                   attempts=attempts+1,
                   finished_at=?,
                   last_error=COALESCE(last_error,'') || ' [orphaned by server startup; max attempts]'
             WHERE session_id IN ({placeholders})
               AND status IN ({status_placeholders})
               AND attempts + 1 >= ?""",
        (datetime.now(UTC).isoformat(), *session_ids, *active_statuses, max_attempts),
    )
    dead = cur.rowcount

    cur = await conn.execute(
        f"""UPDATE tasks
               SET status='pending',
                   lease_owner=NULL,
                   lease_expires_at=NULL,
                   started_at=NULL,
                   attempts=attempts+1,
                   last_error=COALESCE(last_error,'') || ' [orphaned by server startup]'
             WHERE session_id IN ({placeholders})
               AND status IN ({status_placeholders})""",
        (*session_ids, *active_statuses),
    )
    requeued = cur.rowcount
    await conn.commit()
    return {"requeued": requeued, "dead": dead}


async def _citation_text_stats(cfg: Config, session_id: str) -> dict[str, int]:
    return await asyncio.to_thread(_citation_text_stats_sync, cfg, session_id)


def _citation_text_stats_sync(cfg: Config, session_id: str) -> dict[str, int]:
    session_dir = cfg.session_artifact_dir(session_id)
    seen_keys: set[str] = set()
    unique_texts = 0
    total_chars = 0

    def add_record(payload: dict[str, Any], *, fallback_key: str) -> None:
        nonlocal unique_texts, total_chars
        text = _source_record_text(payload)
        if text is None:
            return
        key = _source_record_key(payload, fallback_key=fallback_key, text=text)
        if key in seen_keys:
            return
        seen_keys.add(key)
        unique_texts += 1
        total_chars += len(text)

    if session_dir.is_dir():
        papers_dir = session_dir / "papers"
        if papers_dir.is_dir():
            for path in sorted(papers_dir.glob("*.json")):
                payload = _load_json_object(path)
                if payload is not None:
                    add_record(payload, fallback_key=f"paper:{path.stem}")

        searches_dir = session_dir / "searches"
        if searches_dir.is_dir():
            for path in sorted(searches_dir.glob("*/*.json")):
                payload = _load_json_object(path)
                if payload is None:
                    continue
                for idx, record in enumerate(_iter_source_records(payload)):
                    add_record(record, fallback_key=f"search:{path.stem}:{idx}")

    pdf_stats = _rag_pdf_download_stats_sync(cfg, session_id)
    return {"unique_texts": unique_texts, "total_chars": total_chars, **pdf_stats}


def _rag_pdf_download_stats_sync(cfg: Config, session_id: str) -> dict[str, int]:
    stats = {
        "pdfs_total": 0,
        "pdfs_arxiv": 0,
        "pdfs_biorxiv": 0,
        "pdfs_chemrxiv": 0,
        "pdfs_europe_pmc": 0,
        "pdfs_other": 0,
        "kb_chunks": 0,
        "seed_kb_chunks": 0,
        "kb_seeded": 0,
    }
    rag_dir = cfg.session_rag_dir(session_id)
    manifest = _load_json_object(rag_dir / "manifest.json")
    if manifest is None:
        return stats
    seed = manifest.get("seed") if isinstance(manifest.get("seed"), dict) else {}
    seed_kb = seed.get("kb") if isinstance(seed.get("kb"), dict) else {}
    current_kb = manifest.get("kb") if isinstance(manifest.get("kb"), dict) else {}
    seed_chunks_raw = seed_kb.get("chunk_count")
    current_chunks_raw = current_kb.get("chunk_count")
    seed_chunks = seed_chunks_raw if isinstance(seed_chunks_raw, int) else 0
    current_chunks = current_chunks_raw if isinstance(current_chunks_raw, int) else 0
    stats["seed_kb_chunks"] = max(0, seed_chunks)
    stats["kb_chunks"] = max(0, current_chunks, seed_chunks)
    stats["kb_seeded"] = int(bool(seed) and seed_chunks > 0)

    papers = manifest.get("papers")
    if not isinstance(papers, dict):
        return stats

    pdf_dir = rag_dir / "pdfs"
    seen_files: set[str] = set()
    for item in papers.values():
        if not isinstance(item, dict):
            continue
        file_name = item.get("file")
        if not isinstance(file_name, str) or not file_name.strip():
            continue
        if file_name in seen_files:
            continue
        if not (pdf_dir / file_name).is_file():
            continue
        seen_files.add(file_name)
        stats["pdfs_total"] += 1
        provider = _pdf_provider_from_url(str(item.get("url") or ""))
        stats[f"pdfs_{provider}"] += 1
    return stats


def _pdf_provider_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "arxiv.org" in host:
        return "arxiv"
    if "biorxiv.org" in host and path.startswith("/content/"):
        return "biorxiv"
    if "chemrxiv" in host or ("cambridge.org" in host and "/engage/" in path):
        return "chemrxiv"
    if (
        "europepmc.org" in host
        or "ebi.ac.uk" in host
        or "pubmed.ncbi.nlm.nih.gov" in host
        or "pmc.ncbi.nlm.nih.gov" in host
        or host == "ncbi.nlm.nih.gov"
    ):
        return "europe_pmc"
    return "other"


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _iter_source_records(payload: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            records.extend(_iter_source_records(item))
        return records
    if not isinstance(payload, dict):
        return records
    if _source_record_text(payload) is not None:
        records.append(payload)
    for key in ("results", "records", "items", "papers", "documents"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                records.extend(_iter_source_records(item))
    return records


def _source_record_text(payload: dict[str, Any]) -> str | None:
    for field in ("text", "summary", "abstract", "snippet"):
        value = payload.get(field)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _source_record_key(payload: dict[str, Any], *, fallback_key: str, text: str) -> str:
    for field in ("url", "pdf_url", "abs_url", "doi", "pmid", "arxiv_id", "id", "title"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return f"{field}:{value.strip().lower()}"
    return f"{fallback_key}:text:{ids.text_hash(text)}"


def _sse_event(ev: dict[str, Any]) -> dict[str, str]:
    return {
        "event": str(ev["event"]),
        "data": json.dumps(
            {
                "id": ev["id"],
                "payload": ev["payload"],
                "ts": ev["ts"],
            }
        ),
    }


async def _list_sessions(cfg: Config) -> list[dict[str, Any]]:
    conn = await db_mod.connect(cfg)
    try:
        async with conn.execute(
            """SELECT id, status, research_goal, created_at, updated_at,
                      budget_usd, budget_used_usd,
                      (SELECT COUNT(*) FROM hypotheses WHERE session_id = s.id) AS n_hyps,
                      (SELECT MAX(elo) FROM hypotheses WHERE session_id = s.id) AS top_elo
                 FROM sessions s
                 ORDER BY updated_at DESC LIMIT 50""",
        ) as cur:
            rows = await cur.fetchall()
    finally:
        await conn.close()
    return [dict(r) for r in rows]


async def _recent_matches(conn, session_id: str, *, limit: int) -> list[dict[str, Any]]:
    async with conn.execute(
        """SELECT id, hyp_a, hyp_b, mode, winner, elo_a_after, elo_b_after, created_at
              FROM tournament_matches
             WHERE session_id=?
             ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit),
    ) as cur:
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def _persist_and_publish(
    conn,
    session_id: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    await events_repo.emit(
        conn,
        session_id=session_id,
        task_id=None,
        agent="web",
        event=event,
        payload=payload,
    )
    await GLOBAL_BUS.publish(session_id, event, payload)
