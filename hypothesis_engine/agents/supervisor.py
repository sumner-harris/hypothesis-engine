# Modified from the original work.
"""Supervisor — durable task scheduler for the multi-agent system.

Responsibilities:
1. Parse the scientist's goal into a ResearchPlan.
2. Bootstrap the session (insert row, reclaim expired leases on resume).
3. Run a bounded asyncio worker pool that claims tasks from the DB-backed queue.
4. Apply follow-up scheduling rules after each task completes.
5. Periodically run `decide_next_steps` when the queue is idle:
   - Tournament refinement.
   - Evolution if the leaderboard is stable.
   - Periodic system-feedback meta-reviews.
6. Check the termination predicate after every task; on stop, cancel pending
   work and run a single final meta-review for the overview.
7. Honor pause / abort via DB-flagged session.status.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from .. import ids
from ..citations import render_citations_md
from ..config import Config
from ..llm.anthropic_client import (
    AgentCallSpec,
    CachedBlock,
    CallContext,
)
from ..llm.budgets import TokenBudget
from ..llm.prompts import render
from ..llm.provider import get_provider
from ..llm.routing import route
from ..logging import bind, get_logger
from ..models import ResearchPlan, Session, Task, TaskResult
from ..orchestrator.events import GLOBAL_BUS
from ..orchestrator.feedback_actions import apply_human_feedback_actions
from ..orchestrator.termination import (
    StabilityTracker,
    StopReason,
    should_stop,
    snapshot_top_k,
)
from ..storage import db as db_mod
from ..storage.artifacts import write_text
from ..storage.repos import events as events_repo
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import tasks as task_repo
from ..tools.rag import initialize_session_rag
from ..tools.registry import ToolRegistry
from .base import AgentDeps
from .generation import GenerationAgent, initial_discovery_profile_for_index
from .ranking import RankingAgent
from .reflection import ReflectionAgent
from .schemas import RECORD_RESEARCH_PLAN_TOOL

log = get_logger("supervisor")


def _drain_finished_inflight(inflight: set[asyncio.Task]) -> int:
    """Remove completed worker futures so idle detection sees true idleness."""
    finished = {task for task in inflight if task.done()}
    if not finished:
        return 0
    for task in finished:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except (
            Exception
        ) as exc:  # pragma: no cover - task errors are normally handled inside workers.
            log.exception("worker_task_unhandled", err=str(exc))
    inflight.difference_update(finished)
    return len(finished)


def _initial_generation_payload(
    index: int, total: int, cfg: Config | None = None
) -> dict[str, Any]:
    return {
        "strategy": "literature",
        "n": 1,
        "initial_generation": True,
        "initial_index": index,
        "initial_total": total,
        "discovery_group": "initial",
        "discovery_profile": initial_discovery_profile_for_index(index, cfg),
    }


# ----------------------------- public API ----------------------------- #


class Supervisor:
    """One-process Supervisor; CLI invokes via `await supervisor.run_session(...)`."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    async def run_session(
        self,
        goal: str,
        *,
        preferences_text: str | None = None,
        n_initial: int | None = None,
        wall_clock_seconds: int | None = None,
        resume_session_id: str | None = None,
    ) -> str:
        if n_initial is None:
            n_initial = self.cfg.run.initial_generations
        n_initial = max(1, int(n_initial))
        if resume_session_id is None and int(self.cfg.run.concurrency) < n_initial:
            log.info(
                "initial_generation_concurrency_raised",
                configured_concurrency=self.cfg.run.concurrency,
                n_initial=n_initial,
            )
            self.cfg.run.concurrency = n_initial
        # Capability parsing, cross-reference checks, and executable-tool
        # validation must finish before a database connection or session row
        # is created. ToolRegistry validates after all science skills load.
        tools = ToolRegistry(self.cfg).discover()
        conn = await db_mod.connect(self.cfg)
        try:
            if resume_session_id is None:
                new_session_id = ids.session_id()
                rag_init = await initialize_session_rag(self.cfg, new_session_id)
                session = await self._create_session(
                    conn,
                    goal,
                    preferences_text,
                    wall_clock_seconds,
                    session_id=new_session_id,
                )
                bind(session_id=session.id)
                log.info(
                    "session_started",
                    goal=goal[:120],
                    session_id=session.id,
                    budget_usd=session.budget_usd,
                    n_initial=n_initial,
                )
                await self._emit(
                    conn,
                    session.id,
                    "session_started",
                    {
                        "goal": goal[:200],
                        "n_initial": n_initial,
                        "budget_usd": session.budget_usd,
                        "rag_seed": rag_init,
                    },
                )
                budget = TokenBudget(
                    cfg=self.cfg,
                    budget_tokens=session.budget_tokens,
                    budget_usd=session.budget_usd,
                )
                llm = get_provider(self.cfg, db=conn, budget=budget)
                deps = AgentDeps(cfg=self.cfg, db=conn, llm=llm, tools=tools)

                plan = await self._parse_goal(deps, session, goal, preferences_text)
                await self._apply_plan(conn, session, plan)
                session = await sess_repo.fetch(conn, session.id)
                assert session is not None

                for i in range(n_initial):
                    await task_repo.enqueue(
                        conn,
                        Task(
                            id=ids.task_id(),
                            session_id=session.id,
                            created_at=datetime.now(UTC),
                            agent="generation",
                            action="CreateInitialHypotheses",
                            payload=_initial_generation_payload(i, n_initial, self.cfg),
                            priority=100,
                            status="pending",
                            idempotency_key=f"{session.id}::generation::initial::{i}",
                        ),
                    )
            else:
                session = await sess_repo.fetch(conn, resume_session_id)
                if session is None:
                    raise RuntimeError(f"no such session: {resume_session_id}")
                bind(session_id=session.id)
                rag_init = await initialize_session_rag(self.cfg, session.id)
                log.info("session_resumed", session_id=session.id, status=session.status)
                reclaimed = await task_repo.reclaim_expired_leases(
                    conn,
                    session.id,
                    max_attempts=self.cfg.lease.max_attempts,
                )
                log.info("leases_reclaimed", **reclaimed)
                if session.status != "running":
                    await sess_repo.set_status(conn, session.id, "running")
                    session = await sess_repo.fetch(conn, session.id)
                    assert session is not None
                budget = TokenBudget(
                    cfg=self.cfg,
                    budget_tokens=session.budget_tokens,
                    budget_usd=session.budget_usd,
                )
                llm = get_provider(self.cfg, db=conn, budget=budget)
                deps = AgentDeps(cfg=self.cfg, db=conn, llm=llm, tools=tools)

            termination_min_ideas = int(self.cfg.termination.min_ideas_before_stable)
            if int(self.cfg.run.max_ideas) > 0:
                termination_min_ideas = max(termination_min_ideas, int(self.cfg.run.max_ideas))
            tracker = StabilityTracker(
                k=self.cfg.termination.elo_stability_k,
                n=self.cfg.termination.elo_stability_n,
                eps=self.cfg.termination.elo_stability_eps,
                min_ideas=termination_min_ideas,
                min_matches=self.cfg.termination.min_matches_before_stable,
            )

            stop_reason = await self._main_loop(conn, deps, session, tracker, budget)
            log.info("main_loop_exit", stop_reason=stop_reason.value if stop_reason else "none")

            await self._finalize(conn, deps, session, stop_reason)
            return session.id
        finally:
            await conn.close()

    # ----------------------------- session bootstrap ----------------------------- #

    async def _create_session(
        self,
        conn: aiosqlite.Connection,
        goal: str,
        preferences_text: str | None,
        wall_clock_seconds: int | None,
        session_id: str | None = None,
    ) -> Session:
        sid = session_id or ids.session_id()
        now = datetime.now(UTC)
        wall = wall_clock_seconds or self.cfg.run.wall_clock_seconds
        from datetime import timedelta

        plan = ResearchPlan(objective=goal.strip(), preferences=[], idea_attributes=[])
        snap: dict[str, Any] = json.loads(json.dumps(self.cfg.model_dump(exclude={"secrets"})))
        s = Session(
            id=sid,
            created_at=now,
            updated_at=now,
            status="running",
            research_goal=goal,
            research_plan=plan,
            config_snapshot=snap,
            budget_tokens=self.cfg.run.budget_tokens,
            budget_usd=self.cfg.run.budget_usd,
            wall_deadline=now + timedelta(seconds=wall),
        )
        await sess_repo.insert(conn, s)
        if preferences_text:
            await fb_repo.insert(conn, _human_preference(s.id, preferences_text))
        return s

    async def _parse_goal(
        self,
        deps: AgentDeps,
        session: Session,
        goal: str,
        preferences_text: str | None,
    ) -> ResearchPlan:
        prompt = render(
            "parse_goal",
            goal=goal,
            preferences_text=preferences_text or "",
        )
        r = route(self.cfg, "parse_goal", None)
        spec = AgentCallSpec(
            route=r,
            system_blocks=[
                CachedBlock("You parse research goals into structured plans.", cache=True)
            ],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[RECORD_RESEARCH_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "record_research_plan"},
            max_output_tokens=32_000,
        )
        ctx = CallContext(
            session_id=session.id,
            task_id=None,
            agent="parse_goal",
            action="parse_goal",
            mode=None,
        )
        resp = await deps.llm.call(spec, ctx)
        record: dict[str, Any] | None = None
        for b in resp.raw.content:
            if (
                getattr(b, "type", None) == "tool_use"
                and getattr(b, "name", "") == "record_research_plan"
            ):
                inp = getattr(b, "input", None)
                if isinstance(inp, dict):
                    record = inp
                    break
        if record is None:
            log.warning("parse_goal_no_record", note="falling back to bare ResearchPlan")
            return ResearchPlan(objective=goal.strip(), preferences=[], idea_attributes=[])
        return ResearchPlan(
            objective=record.get("objective", goal.strip()),
            preferences=record.get("preferences", []),
            constraints=record.get("constraints", []),
            idea_attributes=record.get("idea_attributes", []),
            domain_hint=record.get("domain_hint") or None,
            notes=record.get("notes") or None,
        )

    async def _apply_plan(
        self, conn: aiosqlite.Connection, session: Session, plan: ResearchPlan
    ) -> None:
        await conn.execute(
            "UPDATE sessions SET research_plan=?, updated_at=? WHERE id=?",
            (plan.model_dump_json(), datetime.now(UTC).isoformat(), session.id),
        )
        await conn.commit()

    # ----------------------------- main loop ----------------------------- #

    async def _main_loop(
        self,
        conn: aiosqlite.Connection,
        deps: AgentDeps,
        session: Session,
        tracker: StabilityTracker,
        budget: TokenBudget,
    ) -> StopReason | None:
        sem = asyncio.Semaphore(self.cfg.run.concurrency)
        inflight: set[asyncio.Task] = set()
        worker_seq = 0
        last_decide_at = 0.0
        last_reflection_catchup_at = 0.0
        last_snapshot_match_count = -1

        async def _run_task(t: Task) -> None:
            bind(session_id=session.id, task_id=t.id, agent=t.agent)
            async with sem:
                control_conn = await db_mod.connect(self.cfg)
                agent_conn = await db_mod.connect(self.cfg)
                try:
                    task_deps = AgentDeps(
                        cfg=self.cfg,
                        db=agent_conn,
                        llm=get_provider(self.cfg, db=agent_conn, budget=budget),
                        tools=deps.tools,
                    )
                    agent = self._build_agents(task_deps).get(t.agent)

                    await task_repo.mark_in_progress(control_conn, t.id)
                    await self._emit(
                        control_conn,
                        session.id,
                        "task_started",
                        {
                            "task_id": t.id,
                            "agent": t.agent,
                            "action": t.action,
                            "target": t.target_id,
                        },
                    )
                    if agent is None:
                        await task_repo.fail(
                            control_conn,
                            t.id,
                            error=f"no agent: {t.agent}",
                            max_attempts=self.cfg.lease.max_attempts,
                        )
                        return
                    if await self._idea_creation_task_blocked_by_cap(control_conn, t):
                        result = TaskResult(
                            kind="noop",
                            extra={
                                "reason": "max_ideas_reached",
                                "max_ideas": self.cfg.run.max_ideas,
                            },
                        )
                    else:
                        try:
                            result = await self._execute_with_heartbeat(control_conn, t, agent)
                        except Exception as e:
                            await task_repo.fail(
                                control_conn,
                                t.id,
                                error=str(e),
                                max_attempts=self.cfg.lease.max_attempts,
                            )
                            log.exception("task_failed", err=str(e), task_id=t.id, action=t.action)
                            await self._emit(
                                control_conn,
                                session.id,
                                "task_failed",
                                {
                                    "task_id": t.id,
                                    "agent": t.agent,
                                    "action": t.action,
                                    "target": t.target_id,
                                    "err": str(e)[:300],
                                },
                            )
                            return
                    await self._apply_follow_ups(control_conn, session, t, result)
                    await task_repo.complete(control_conn, t.id)
                    completion_payload = {
                        "task_id": t.id,
                        "agent": t.agent,
                        "action": t.action,
                        "kind": result.kind,
                        "follow_hypothesis_ids": result.hypothesis_ids[:5],
                        "match_ids": result.match_ids[:5],
                        "extra": result.extra,
                    }
                    await self._emit(control_conn, session.id, "task_completed", completion_payload)
                    reason = result.extra.get("reason") if isinstance(result.extra, dict) else None
                    if result.kind == "noop" and reason:
                        await self._emit(
                            control_conn,
                            session.id,
                            "task_warning",
                            {
                                "task_id": t.id,
                                "agent": t.agent,
                                "action": t.action,
                                "reason": reason,
                                "extra": result.extra,
                            },
                        )
                finally:
                    await agent_conn.close()
                    await control_conn.close()

        try:
            while True:
                # Check external pause/abort by re-reading session status.
                refreshed = await sess_repo.fetch(conn, session.id)
                external_stop = refreshed is not None and refreshed.status in ("aborted",)
                if refreshed is not None and refreshed.status == "paused":
                    # Wait until unpaused (or aborted).
                    await asyncio.sleep(1.0)
                    continue

                if inflight:
                    _drain_finished_inflight(inflight)

                # Termination check (refreshes budget_used_* from the row)
                if refreshed is not None:
                    stop = should_stop(self.cfg, refreshed, tracker, external_stop=external_stop)
                    if (
                        stop is StopReason.ELO_STABLE
                        and await self._defer_elo_stable_stop_for_evolution(conn, session.id)
                    ):
                        stop = None
                    if stop is not None:
                        # Wait for inflight to drain before returning.
                        if inflight:
                            await asyncio.wait(inflight)
                        return stop

                # Catch drafts that were persisted by long-running tasks before
                # their normal task-completion follow-ups have fired.
                now = time.monotonic()
                if now - last_reflection_catchup_at >= 10.0:
                    last_reflection_catchup_at = now
                    await self._enqueue_missing_reflections_for_drafts(conn, session.id)

                if await self._wait_for_rag_ingest_if_needed(conn, session.id):
                    continue

                # Refill worker slots.
                slots_open = self.cfg.run.concurrency - len(inflight)
                claimed: list[Task] = []
                for _ in range(slots_open):
                    t = await task_repo.claim_one(
                        conn,
                        session.id,
                        worker_id=f"w{worker_seq}",
                        lease_seconds=self.cfg.lease.default_seconds,
                    )
                    if t is None:
                        break
                    worker_seq += 1
                    claimed.append(t)
                for t in claimed:
                    inflight.add(asyncio.create_task(_run_task(t)))

                # Update stability snapshot when match count crossed the threshold.
                snap = await snapshot_top_k(conn, session.id, self.cfg.termination.elo_stability_k)
                if (
                    snap.match_count
                    >= last_snapshot_match_count + self.cfg.termination.match_snapshot_every
                ):
                    tracker.push(snap)
                    last_snapshot_match_count = snap.match_count
                    log.info(
                        "elo_snapshot",
                        match_count=snap.match_count,
                        top_ids=list(snap.top_ids),
                        top_elos=list(snap.top_elos),
                    )

                # If the queue is empty and worker slots are open, run
                # decide_next_steps at most every ~10s. This also covers the
                # common long-tail case where one reflection/evolution call is
                # still running but the rest of the worker pool could keep
                # ranking/refinement moving.
                if slots_open > 0 and not claimed:
                    pending = await task_repo.count_by_status(conn, session.id)
                    if pending.get("pending", 0) == 0:
                        now = time.monotonic()
                        if now - last_decide_at >= 10.0:
                            last_decide_at = now
                            scheduled = await self._decide_next_steps(
                                conn, session, tracker=tracker
                            )
                            if scheduled == 0:
                                if inflight:
                                    _drain_finished_inflight(inflight)
                                if not inflight:
                                    # truly idle and no progress possible — exit gracefully
                                    return StopReason.IDLE
                                await asyncio.sleep(1.0)
                                continue
                            continue
                        # Wait briefly so we don't spin while throttling idle
                        # scheduling; do not block on a single long tail task.
                        await asyncio.sleep(1.0)
                        continue

                if not inflight:
                    # Nothing claimed AND nothing running — but tasks may be pending
                    # in other workers' future claims; brief sleep and retry.
                    await asyncio.sleep(0.1)
                    continue

                _done, pending = await asyncio.wait(inflight, return_when=asyncio.FIRST_COMPLETED)
                inflight = set(pending)
        finally:
            if inflight:
                # Best effort: let any inflight task finish before returning.
                await asyncio.wait(inflight)

    async def _idea_creation_task_blocked_by_cap(
        self, conn: aiosqlite.Connection, task: Task
    ) -> bool:
        if task.agent not in {"generation", "evolution"}:
            return False
        max_ideas = int(self.cfg.run.max_ideas)
        if max_ideas <= 0:
            return False
        current = await hyp_repo.count_for_session(conn, task.session_id)
        return current >= max_ideas

    async def _execute_with_heartbeat(
        self,
        conn: aiosqlite.Connection,
        task: Task,
        agent,
    ):
        work = asyncio.create_task(agent.execute(task))
        while True:
            done, _ = await asyncio.wait(
                {work},
                timeout=self.cfg.lease.heartbeat_seconds,
            )
            if work in done:
                return work.result()
            await task_repo.heartbeat(
                conn,
                task.id,
                lease_seconds=self.cfg.lease.default_seconds,
            )

    async def _wait_for_rag_ingest_if_needed(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
    ) -> bool:
        if not await self._rag_ingest_wait_needed(conn, session_id):
            return False

        from ..tools.rag import (
            background_ingest_status,
            wait_for_background_ingest_step,
        )

        started = time.monotonic()
        status = background_ingest_status(self.cfg, session_id)
        min_indexed = max(
            0,
            int(self.cfg.rag.generation_wait_min_indexed_papers or 0),
        )
        log.info(
            "rag_ingest_wait_started",
            active_background_tasks=status.get("active_background_tasks", 0),
            reserved_paper_count=status.get("reserved_paper_count", 0),
            unindexed_paper_count=status.get("unindexed_paper_count", 0),
        )
        await self._emit(conn, session_id, "rag_ingest_wait_started", status)

        while True:
            refreshed = await sess_repo.fetch(conn, session_id)
            if refreshed is not None and refreshed.status in ("paused", "aborted"):
                return True

            status = await wait_for_background_ingest_step(
                self.cfg,
                session_id,
                timeout_seconds=1.0,
            )
            seed_kb_ready = status.get("seed_kb_ready") is True
            indexed = int(status.get("indexed_paper_count") or 0)
            if seed_kb_ready or (min_indexed > 0 and indexed >= min_indexed):
                waited_ms = int((time.monotonic() - started) * 1000)
                release_reason = "seed_kb_ready" if seed_kb_ready else "minimum_indexed_papers"
                status = {
                    **status,
                    "waited_ms": waited_ms,
                    "released_early": True,
                    "release_reason": release_reason,
                    "generation_wait_min_indexed_papers": min_indexed,
                }
                log.info(
                    "rag_ingest_wait_completed",
                    waited_ms=waited_ms,
                    seed_chunk_count=status.get("seed_chunk_count", 0),
                    released_early=True,
                    release_reason=release_reason,
                )
                await self._emit(conn, session_id, "rag_ingest_wait_completed", status)
                return True
            if not status.get("pending_background_ingest"):
                waited_ms = int((time.monotonic() - started) * 1000)
                status = {**status, "waited_ms": waited_ms}
                log.info(
                    "rag_ingest_wait_completed",
                    waited_ms=waited_ms,
                    paper_count=status.get("paper_count", 0),
                    indexed_paper_count=status.get("indexed_paper_count", 0),
                    reserved_paper_count=status.get("reserved_paper_count", 0),
                    unindexed_paper_count=status.get("unindexed_paper_count", 0),
                )
                await self._emit(conn, session_id, "rag_ingest_wait_completed", status)
                return True

    async def _rag_ingest_wait_needed(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
    ) -> bool:
        if not self.cfg.rag.enabled:
            return False
        try:
            from ..tools.rag import background_ingest_pending
        except ImportError:
            return False
        if not background_ingest_pending(self.cfg, session_id):
            return False

        async with conn.execute(
            """SELECT agent, COUNT(*) AS n
                  FROM tasks
                 WHERE session_id=? AND status='pending'
                 GROUP BY agent""",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        pending_by_agent = {row["agent"]: int(row["n"] or 0) for row in rows}
        return not (pending_by_agent and set(pending_by_agent) <= {"generation"})

    # ----------------------------- follow-up rules ----------------------------- #

    async def _enqueue_missing_reflections_for_drafts(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
    ) -> int:
        async with conn.execute(
            """SELECT h.id
                  FROM hypotheses h
             LEFT JOIN reviews r
                    ON r.hypothesis_id=h.id AND r.kind='full'
             LEFT JOIN tasks t
                    ON t.idempotency_key=h.id || '::review::full'
                 WHERE h.session_id=?
                   AND h.state='draft'
                   AND r.id IS NULL
                   AND t.id IS NULL
                 ORDER BY h.created_at""",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()

        enqueued = 0
        for row in rows:
            hid = row["id"]
            inserted = await task_repo.enqueue(
                conn,
                Task(
                    id=ids.task_id(),
                    session_id=session_id,
                    created_at=datetime.now(UTC),
                    agent="reflection",
                    action="ReviewHypothesis",
                    target_id=hid,
                    payload={"kind": "full"},
                    priority=100,
                    status="pending",
                    idempotency_key=f"{hid}::review::full",
                ),
            )
            if inserted:
                enqueued += 1
        return enqueued

    async def _apply_follow_ups(
        self,
        conn: aiosqlite.Connection,
        session: Session,
        task: Task,
        result,
    ) -> None:
        if result.kind == "hypothesis_created":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="reflection",
                        action="ReviewHypothesis",
                        target_id=hid,
                        payload={"kind": "full"},
                        priority=100,
                        status="pending",
                        idempotency_key=f"{hid}::review::full",
                    ),
                )
        elif result.kind == "review_completed":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="ranking",
                        action="AddToTournament",
                        target_id=hid,
                        payload={},
                        priority=80,
                        status="pending",
                        idempotency_key=f"{hid}::ranking::add",
                    ),
                )
                await self._enqueue_pin_exploration(
                    conn, session.id, hid, key_suffix=f"review::{task.id}"
                )
        elif result.kind == "added_to_tournament":
            for hid in result.hypothesis_ids:
                await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="ranking",
                        action="RunTournamentBatch",
                        target_id=None,
                        payload={"focus": hid},
                        priority=120,
                        status="pending",
                        idempotency_key=f"{hid}::ranking::focus_batch",
                    ),
                )
                await self._enqueue_pin_exploration(
                    conn, session.id, hid, key_suffix=f"tournament::{task.id}"
                )
        elif result.kind == "tournament_match_complete":
            n_matches = result.extra.get("total_matches_after")
            _ = n_matches
            # Periodically re-cluster the proximity graph.
            from ..storage.repos import tournaments as tourney_repo

            mc = await tourney_repo.count_matches(conn, session.id)
            if mc > 0 and mc % self.cfg.vectors.full_recluster_every_matches == 0:
                await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="proximity",
                        action="UpdateProximityGraph",
                        target_id=None,
                        payload={"rebuild": True},
                        priority=200,
                        status="pending",
                        idempotency_key=f"{session.id}::proximity::{mc}",
                    ),
                )
            await self._enqueue_due_metareview(conn, session.id, match_count=mc)
            await self._enqueue_due_top_refinement(conn, session, match_count=mc)

    async def _enqueue_due_metareview(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        *,
        match_count: int | None = None,
    ) -> bool:
        """Enqueue periodic system feedback once per 50 valid tournament matches."""
        from ..storage.repos import tournaments as tourney_repo

        mc = match_count
        if mc is None:
            mc = await tourney_repo.count_matches(conn, session_id)
        async with conn.execute(
            """SELECT COUNT(*) AS n FROM system_feedback
                  WHERE session_id=? AND kind='system_feedback' AND source='meta_review'""",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        feedback_count = row["n"] if row else 0
        next_feedback_index = feedback_count + 1
        if mc < next_feedback_index * 50:
            return False

        return await task_repo.enqueue(
            conn,
            Task(
                id=ids.task_id(),
                session_id=session_id,
                created_at=datetime.now(UTC),
                agent="metareview",
                action="GenerateSystemFeedback",
                target_id=None,
                payload={
                    "reason": "periodic_tournament_threshold",
                    "match_count": mc,
                    "feedback_index": next_feedback_index,
                },
                priority=95,
                status="pending",
                idempotency_key=f"{session_id}::metareview::feedback::{next_feedback_index}",
            ),
        )

    async def _enqueue_due_top_refinement(
        self,
        conn: aiosqlite.Connection,
        session: Session,
        *,
        match_count: int,
    ) -> int:
        """Periodically keep leaderboard hypotheses challenged despite backlog."""
        from ..storage.repos import tournaments as tourney_repo

        interval = max(1, int(self.cfg.ranking.idle_parallel_tasks))
        if match_count <= 0 or match_count % interval != 0:
            return 0

        in_tournament = await hyp_repo.tournament_candidates(conn, session.id)
        if len(in_tournament) < 2:
            return 0
        available_pairs = await tourney_repo.eligible_pair_count(
            conn,
            session.id,
            [h.id for h in in_tournament],
            max_pair_matches=int(self.cfg.ranking.pair_max_matches),
            wins_to_close_pair=int(self.cfg.ranking.pair_wins_to_close),
            exclude_active=True,
        )
        if available_pairs <= 0:
            return 0
        mature_match_count = max(1, int(self.cfg.evolution.mature_matches))
        mature_hypotheses = [h for h in in_tournament if h.matches_played >= mature_match_count]
        if len(mature_hypotheses) < self.cfg.evolution.min_mature:
            return 0

        rankable = list(in_tournament)

        top_quota = max(1, interval // 3)
        top_ranked = sorted(
            rankable,
            key=lambda h: (
                -(h.elo or self.cfg.ranking.elo_initial),
                h.matches_played,
                h.id,
            ),
        )[:top_quota]
        bucket = match_count // interval
        enqueued = 0
        for h in top_ranked:
            inserted = await task_repo.enqueue(
                conn,
                Task(
                    id=ids.task_id(),
                    session_id=session.id,
                    created_at=datetime.now(UTC),
                    agent="ranking",
                    action="RunTournamentBatch",
                    target_id=None,
                    payload={"focus": h.id, "reason": "top_refine"},
                    priority=98,
                    status="pending",
                    idempotency_key=(f"{session.id}::ranking::top_refine::{bucket}::{h.id}"),
                ),
            )
            if inserted:
                enqueued += 1
        return enqueued

    async def _enqueue_pin_exploration(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        hypothesis_id: str,
        *,
        key_suffix: str,
    ) -> int:
        h = await hyp_repo.fetch(conn, hypothesis_id)
        if h is None or h.session_id != session_id or h.state != "pinned" or h.elo is None:
            return 0
        reviews = await rev_repo.list_for_hypothesis(conn, hypothesis_id)
        has_complete_review = any(
            r.kind == "full"
            and r.verdict is not None
            and r.scores.novelty is not None
            and r.scores.correctness is not None
            and r.scores.testability is not None
            and r.scores.feasibility is not None
            for r in reviews
        )
        if not has_complete_review:
            return 0
        enqueued = 0
        inserted = await task_repo.enqueue(
            conn,
            Task(
                id=ids.task_id(),
                session_id=session_id,
                created_at=datetime.now(UTC),
                agent="ranking",
                action="RunTournamentBatch",
                target_id=None,
                payload={"focus": hypothesis_id, "reason": "human_pin"},
                priority=60,
                status="pending",
                idempotency_key=f"{hypothesis_id}::ranking::focus::pin::{key_suffix}",
            ),
        )
        if inserted:
            enqueued += 1
        if not await self._idea_cap_reached(conn, session_id):
            inserted = await task_repo.enqueue(
                conn,
                Task(
                    id=ids.task_id(),
                    session_id=session_id,
                    created_at=datetime.now(UTC),
                    agent="evolution",
                    action="EvolveTopHypotheses",
                    target_id=hypothesis_id,
                    payload={
                        "focus": hypothesis_id,
                        "top_k": 6,
                        "strategies": ["simplify", "feasibility", "out_of_box"],
                        "reason": "human_pin",
                    },
                    priority=90,
                    status="pending",
                    idempotency_key=f"{hypothesis_id}::evolution::pin::{key_suffix}",
                ),
            )
            if inserted:
                enqueued += 1
        return enqueued

    async def _idea_cap_reached(self, conn: aiosqlite.Connection, session_id: str) -> bool:
        max_ideas = int(self.cfg.run.max_ideas)
        if max_ideas <= 0:
            return False
        return await hyp_repo.count_for_session(conn, session_id) >= max_ideas

    async def _defer_elo_stable_stop_for_evolution(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> bool:
        max_ideas = int(self.cfg.run.max_ideas)
        if max_ideas <= 0 or not self.cfg.evolution.require_rank_stability:
            return False
        return not await self._idea_cap_reached(conn, session_id)

    def _evolution_rank_stable(self, tracker: StabilityTracker | None) -> bool:
        if not self.cfg.evolution.require_rank_stability:
            return True
        if tracker is None:
            return False
        n = max(1, int(self.cfg.evolution.rank_stability_n))
        return tracker.has_stable_top_set(
            n=n,
            min_ideas=max(0, int(self.cfg.evolution.rank_stability_min_ideas)),
            min_matches=max(0, int(self.cfg.evolution.rank_stability_min_matches)),
        )

    async def _candidate_feed_backlog_count(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> int:
        async with conn.execute(
            """SELECT COUNT(*) AS n
                  FROM tasks
                 WHERE session_id=?
                   AND agent IN ('generation','reflection','evolution')
                   AND status IN ('pending','leased','in_progress')""",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"] if row else 0)

    async def _active_evolution_task_count(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> int:
        async with conn.execute(
            """SELECT COUNT(*) AS n
                  FROM tasks
                 WHERE session_id=?
                   AND agent='evolution'
                   AND action='EvolveTopHypotheses'
                   AND status IN ('pending','leased','in_progress')""",
            (session_id,),
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"] if row else 0)

    async def _last_idle_evolution_anchor(
        self, conn: aiosqlite.Connection, session_id: str
    ) -> int | None:
        prefix = f"{session_id}::evolution::idle::"
        async with conn.execute(
            """SELECT idempotency_key
                  FROM tasks
                 WHERE session_id=?
                   AND agent='evolution'
                   AND action='EvolveTopHypotheses'
                   AND idempotency_key LIKE ?""",
            (session_id, prefix + "%"),
        ) as cur:
            rows = await cur.fetchall()

        anchors: list[int] = []
        for row in rows:
            key = row["idempotency_key"] or ""
            if not key.startswith(prefix):
                continue
            raw = key[len(prefix) :].split("::", 1)[0]
            try:
                anchors.append(int(raw))
            except ValueError:
                continue
        return max(anchors) if anchors else None

    def _small_pool_ranking_throttled(self, in_tournament, feed_backlog: int) -> bool:
        if feed_backlog <= 0 or len(in_tournament) < 2:
            return False
        small_pool_size = max(2, int(self.cfg.ranking.small_pool_backlog_size))
        if len(in_tournament) > small_pool_size:
            return False
        match_cap = max(1, int(self.cfg.ranking.small_pool_backlog_match_cap))
        return all(h.matches_played >= match_cap for h in in_tournament)

    # ----------------------------- decide_next_steps ----------------------------- #

    async def _decide_next_steps(
        self,
        conn: aiosqlite.Connection,
        session: Session,
        *,
        tracker: StabilityTracker | None = None,
    ) -> int:
        """When the queue empties: refill it with refinement work. Returns # enqueued."""
        from ..storage.repos import tournaments as tourney_repo

        enqueued = 0

        # We anchor idle-refinement idempotency keys on the current match count
        # rather than a fresh task id. Otherwise every idle pass — which can
        # fire every ~10s — would enqueue a *new* tournament/evolution task
        # even when a prior one is still pending, flooding the queue and
        # double-counting work toward the budget.
        anchor_mc = await tourney_repo.count_matches(conn, session.id)

        async with conn.execute(
            """SELECT id, kind, target_id FROM system_feedback
                  WHERE session_id=? AND active=1 AND source='human'
                  ORDER BY created_at DESC""",
            (session.id,),
        ) as cur:
            feedback_rows = await cur.fetchall()
        for feedback in feedback_rows:
            actions = await apply_human_feedback_actions(
                conn,
                session_id=session.id,
                feedback_id=feedback["id"],
                kind=feedback["kind"],
                target_id=feedback["target_id"],
            )
            enqueued += int(actions.get("enqueued") or 0)

        # Keep Elo refining. Fan out focused ranking tasks so the pool warms up
        # and later stays balanced instead of repeatedly spending one generic
        # match at a time.
        in_tournament = await hyp_repo.tournament_candidates(conn, session.id)
        mature_match_count = max(1, int(self.cfg.evolution.mature_matches))
        mature_hypotheses = [h for h in in_tournament if h.matches_played >= mature_match_count]
        feed_backlog = await self._candidate_feed_backlog_count(conn, session.id)
        ranking_throttled = self._small_pool_ranking_throttled(in_tournament, feed_backlog)
        if ranking_throttled:
            log.info(
                "ranking_small_pool_throttled",
                tournament_size=len(in_tournament),
                feed_backlog=feed_backlog,
                match_cap=self.cfg.ranking.small_pool_backlog_match_cap,
            )
        candidate_ids = [h.id for h in in_tournament]
        eligible_pairs_total = await tourney_repo.eligible_pair_count(
            conn,
            session.id,
            candidate_ids,
            max_pair_matches=int(self.cfg.ranking.pair_max_matches),
            wins_to_close_pair=int(self.cfg.ranking.pair_wins_to_close),
            exclude_active=False,
        )
        eligible_pairs_available = await tourney_repo.eligible_pair_count(
            conn,
            session.id,
            candidate_ids,
            max_pair_matches=int(self.cfg.ranking.pair_max_matches),
            wins_to_close_pair=int(self.cfg.ranking.pair_wins_to_close),
            exclude_active=True,
        )
        no_eligible_matches = len(in_tournament) >= 2 and eligible_pairs_total == 0
        if len(in_tournament) >= 2 and not ranking_throttled and eligible_pairs_available > 0:
            idle_parallel = max(1, int(self.cfg.ranking.idle_parallel_tasks))
            ranking_jobs: list[tuple[str | None, dict[str, Any], str, int]] = []
            rankable = list(in_tournament)
            if len(mature_hypotheses) < self.cfg.evolution.min_mature:
                non_mature = [h for h in rankable if h.matches_played < mature_match_count]
                non_mature.sort(
                    key=lambda h: (
                        -h.matches_played,
                        -(h.elo or self.cfg.ranking.elo_initial),
                        h.id,
                    )
                )
                focused = non_mature[:idle_parallel]
                ranking_jobs = [
                    (
                        h.id,
                        {"focus": h.id, "reason": "warmup"},
                        f"warmup::{h.id}",
                        150,
                    )
                    for h in focused
                ]
            else:
                rankable = rankable or in_tournament
                ranking_jobs = []
                seen_focus_ids: set[str] = set()

                def add_ranking_job(h, reason: str, priority: int) -> None:
                    if h.id in seen_focus_ids or len(ranking_jobs) >= idle_parallel:
                        return
                    seen_focus_ids.add(h.id)
                    ranking_jobs.append(
                        (
                            h.id,
                            {"focus": h.id, "reason": reason},
                            f"{reason}::{h.id}",
                            priority,
                        )
                    )

                top_quota = max(1, idle_parallel // 3)
                top_ranked = sorted(
                    rankable,
                    key=lambda h: (
                        -(h.elo or self.cfg.ranking.elo_initial),
                        h.matches_played,
                        h.id,
                    ),
                )
                for h in top_ranked[:top_quota]:
                    add_ranking_job(h, "top_refine", 98)

                under_matched = sorted(
                    rankable,
                    key=lambda h: (
                        h.matches_played,
                        -(h.elo or self.cfg.ranking.elo_initial),
                        h.id,
                    ),
                )
                for h in under_matched:
                    add_ranking_job(h, "refine", 150)

            if not ranking_jobs:
                ranking_jobs = [(None, {}, "global", 150)]

            for _focus_id, payload, key_suffix, priority in ranking_jobs:
                inserted = await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="ranking",
                        action="RunTournamentBatch",
                        target_id=None,
                        payload=payload,
                        priority=priority,
                        status="pending",
                        idempotency_key=f"{session.id}::ranking::idle::{anchor_mc}::{key_suffix}",
                    ),
                )
                if inserted:
                    enqueued += 1

        # If the leaderboard has matured, evolve. The maturity gate, top_k, and
        # parallel task count are config-driven so deep runs can keep workers busy
        # during refinement instead of evolving one focused hypothesis per idle pass.
        can_create_more_ideas = not await self._idea_cap_reached(conn, session.id)
        evolution_cadence_ready = False
        evolution_rank_stable = self._evolution_rank_stable(tracker)
        if can_create_more_ideas and len(mature_hypotheses) >= self.cfg.evolution.min_mature:
            active_evolution = await self._active_evolution_task_count(conn, session.id)
            last_evolution_anchor = await self._last_idle_evolution_anchor(conn, session.id)
            min_tournament_matches = max(0, int(self.cfg.evolution.min_tournament_matches))
            min_matches_between_batches = max(
                1, int(self.cfg.evolution.min_matches_between_batches)
            )
            evolution_trigger_ready = evolution_rank_stable or no_eligible_matches
            evolution_cadence_ready = (
                active_evolution == 0
                and evolution_trigger_ready
                and (no_eligible_matches or anchor_mc >= min_tournament_matches)
                and (
                    last_evolution_anchor is None
                    or anchor_mc >= last_evolution_anchor + min_matches_between_batches
                )
            )
            if not evolution_cadence_ready:
                log.info(
                    "evolution_idle_throttled",
                    active_evolution=active_evolution,
                    rank_stable=evolution_rank_stable,
                    no_eligible_matches=no_eligible_matches,
                    eligible_pairs_total=eligible_pairs_total,
                    eligible_pairs_available=eligible_pairs_available,
                    match_count=anchor_mc,
                    last_evolution_anchor=last_evolution_anchor,
                    min_tournament_matches=min_tournament_matches,
                    min_matches_between_batches=min_matches_between_batches,
                )
        if (
            can_create_more_ideas
            and len(mature_hypotheses) >= self.cfg.evolution.min_mature
            and evolution_cadence_ready
        ):
            idle_parallel = max(1, int(self.cfg.evolution.idle_parallel_tasks))
            if idle_parallel == 1:
                evolution_jobs: list[tuple[str | None, list[str], int, str]] = [
                    (
                        None,
                        ["combine", "simplify", "out_of_box"],
                        self.cfg.evolution.top_k,
                        "global",
                    )
                ]
            else:
                evolution_jobs = [(None, ["combine"], self.cfg.evolution.top_k, "combine")]
                for h in mature_hypotheses[: idle_parallel - 1]:
                    evolution_jobs.append(
                        (
                            h.id,
                            ["simplify", "feasibility", "out_of_box"],
                            min(self.cfg.evolution.top_k, 6),
                            f"focus::{h.id}",
                        )
                    )

            for focus_id, strategies, top_k, key_suffix in evolution_jobs:
                payload: dict[str, Any] = {"top_k": top_k, "strategies": strategies}
                if focus_id:
                    payload["focus"] = focus_id
                inserted = await task_repo.enqueue(
                    conn,
                    Task(
                        id=ids.task_id(),
                        session_id=session.id,
                        created_at=datetime.now(UTC),
                        agent="evolution",
                        action="EvolveTopHypotheses",
                        target_id=focus_id,
                        payload=payload,
                        priority=140,
                        status="pending",
                        idempotency_key=f"{session.id}::evolution::idle::{anchor_mc}::{key_suffix}",
                    ),
                )
                if inserted:
                    enqueued += 1

        # Periodic meta-review is also checked on match-completion follow-ups so
        # it cannot be starved by a busy refinement queue. Keep this idle check
        # for sessions that cross a threshold before the follow-up path runs.
        if await self._enqueue_due_metareview(conn, session.id, match_count=anchor_mc):
            enqueued += 1

        return enqueued

    # ----------------------------- finalize ----------------------------- #

    async def _finalize(
        self,
        conn: aiosqlite.Connection,
        deps: AgentDeps,
        session: Session,
        stop_reason: StopReason | None,
    ) -> None:
        n_cancel = await task_repo.cancel_pending_for_session(conn, session.id)
        if n_cancel:
            log.info("pending_cancelled", n=n_cancel)

        # Try to run the proper final overview via metareview if the agent exists.
        # Fall back to the stub if metareview is not yet wired in (older builds).
        try:
            from .metareview import MetaReviewAgent

            agent = MetaReviewAgent(deps)
            final_task = Task(
                id=ids.task_id(),
                session_id=session.id,
                created_at=datetime.now(UTC),
                agent="metareview",
                action="GenerateFinalResearchOverview",
                target_id=None,
                payload={},
                priority=1,
                status="pending",
                idempotency_key=f"{session.id}::metareview::final",
            )
            await task_repo.enqueue(conn, final_task)
            await task_repo.mark_in_progress(conn, final_task.id)
            try:
                result = await agent.execute(final_task)
                overview_path = result.extra.get("overview_path")
                if overview_path:
                    await sess_repo.set_final_overview(conn, session.id, overview_path)
                await task_repo.complete(conn, final_task.id)
            except Exception as e:
                log.exception("final_overview_failed", err=str(e))
                await task_repo.fail(
                    conn, final_task.id, error=str(e), max_attempts=self.cfg.lease.max_attempts
                )
                overview_path = await self._write_simple_overview(conn, session)
                await sess_repo.set_final_overview(conn, session.id, overview_path)
        except ImportError:
            overview_path = await self._write_simple_overview(conn, session)
            await sess_repo.set_final_overview(conn, session.id, overview_path)

        # `set_final_overview` flips status to 'done' atomically. If the
        # overview path was never set (e.g. metareview crashed and the simple
        # overview also failed) the status is still 'running'; force-set it
        # here so the session doesn't appear to be running forever after exit.
        # For EXTERNAL stops we don't overwrite the user-set 'paused' /
        # 'aborted' status.
        if stop_reason != StopReason.EXTERNAL:
            await sess_repo.set_status(conn, session.id, "done")

        await self._emit(
            conn,
            session.id,
            "session_done",
            {"stop_reason": stop_reason.value if stop_reason else None},
        )

    async def _write_simple_overview(self, conn: aiosqlite.Connection, session: Session) -> str:
        hyps = await hyp_repo.list_for_session(conn, session.id)
        parts: list[str] = [
            f"# Research overview — session {session.id}",
            f"\n**Goal.** {session.research_goal}\n",
            f"**Hypotheses produced.** {len(hyps)}",
            "",
        ]
        for i, h in enumerate(hyps, 1):
            parts.append(f"## {i}. {h.title or h.id}")
            parts.append(
                f"`{h.id}` — strategy `{h.strategy}` — state `{h.state}` — Elo `{h.elo:.0f}`"
                if h.elo is not None
                else f"`{h.id}` — strategy `{h.strategy}` — state `{h.state}`"
            )
            parts.append(h.summary or "(no summary)")
            citations_md = render_citations_md(
                h.citations, heading="**Citations:**", include_excerpts=False
            )
            if citations_md:
                parts.append(citations_md)
            reviews = await rev_repo.list_for_hypothesis(conn, h.id)
            if reviews:
                parts.append("\n**Reviews:**")
                for r in reviews:
                    parts.append(
                        f"- *{r.kind}* — verdict `{r.verdict or '?'}` "
                        f"(n={r.scores.novelty}, c={r.scores.correctness}, "
                        f"t={r.scores.testability})"
                    )
            parts.append("")
        body = "\n".join(parts)
        return await write_text(self.cfg, session.id, "final", "overview", ".md", body)

    # ----------------------------- helpers ----------------------------- #

    def _build_agents(self, deps: AgentDeps) -> dict[str, object]:
        out: dict[str, object] = {
            "generation": GenerationAgent(deps),
            "reflection": ReflectionAgent(deps),
            "ranking": RankingAgent(deps),
        }
        # Evolution / Proximity / Meta-review register if importable.
        try:
            from .evolution import EvolutionAgent

            out["evolution"] = EvolutionAgent(deps)
        except ImportError:
            pass
        try:
            from .proximity import ProximityAgent

            out["proximity"] = ProximityAgent(deps)
        except ImportError:
            pass
        try:
            from .metareview import MetaReviewAgent

            out["metareview"] = MetaReviewAgent(deps)
        except ImportError:
            pass
        return out

    async def _emit(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await events_repo.emit(
            conn,
            session_id=session_id,
            task_id=None,
            agent="supervisor",
            event=event,
            payload=payload,
        )
        await GLOBAL_BUS.publish(session_id, event, payload)


# ----------------------------- helpers ----------------------------- #


def _human_preference(session_id: str, text: str):
    from ..models import SystemFeedback

    return SystemFeedback(
        id=ids.feedback_id(),
        session_id=session_id,
        created_at=datetime.now(UTC),
        source="human",
        kind="preference",
        target_id=None,
        text=text,
        active=True,
    )
