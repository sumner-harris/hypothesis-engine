# Modified from the original work.
"""Meta-review agent — periodic system feedback + final research overview.

Two actions:
- `GenerateSystemFeedback`           — Sonnet + thinking; writes a SystemFeedback row.
  The body is auto-injected into future Generation/Evolution prompts via the
  `latest_system_feedback` query the agents already perform.
- `GenerateFinalResearchOverview`    — Opus + max thinking; writes the markdown
  report and updates `sessions.final_overview`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .. import ids
from ..citations import render_hypothesis_citation_appendix
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..logging import get_logger
from ..models import SystemFeedback, Task, TaskResult
from ..storage.artifacts import write_json, write_text
from ..storage.repos import feedback as fb_repo
from ..storage.repos import hypotheses as hyp_repo
from ..storage.repos import reviews as rev_repo
from ..storage.repos import sessions as sess_repo
from ..storage.repos import tournaments as tourney_repo
from .base import BaseAgent
from .schemas import RECORD_SYSTEM_FEEDBACK_TOOL

log = get_logger("metareview")


class MetaReviewAgent(BaseAgent):
    name = "metareview"

    async def execute(self, task: Task) -> TaskResult:
        if task.action == "GenerateSystemFeedback":
            return await self._system_feedback(task)
        if task.action == "GenerateFinalResearchOverview":
            return await self._final_overview(task)
        raise ValueError(f"MetaReviewAgent does not handle action {task.action!r}")

    # ----------------------------- system feedback ----------------------------- #

    async def _system_feedback(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        reviews = await rev_repo.list_for_session(self.deps.db, session.id)
        human_feedback = await fb_repo.active_human_for_session(self.deps.db, session.id)
        if not reviews and not human_feedback:
            return TaskResult(kind="noop", extra={"reason": "no reviews or human feedback yet"})

        reviews_block = (
            "\n\n---\n\n".join(
                f"### Review of `{r.hypothesis_id}` (kind={r.kind}, verdict={r.verdict or '?'})\n{r.body[:3000]}"
                for r in reviews[:50]
            )
            or "(none yet)"
        )
        human_feedback_block = self._human_feedback_block(human_feedback)
        rationales = await tourney_repo.recent_rationales(self.deps.db, session.id, limit=50)
        debate_block = "\n\n---\n\n".join(rat[:1500] for rat in rationales if rat)

        prompt = render(
            "metareview.system",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            instructions=human_feedback_block,
            reviews=reviews_block,
            debate_rationales=debate_block,
        )
        r = route(self.deps.cfg, "metareview", "system")
        spec = AgentCallSpec(
            route=r,
            system_blocks=[
                CachedBlock(self._system_prompt_header(), cache=True),
                CachedBlock(
                    f"# Research goal\n{session.research_goal}\n\n"
                    f"# Preferences\n{'; '.join(session.research_plan.preferences)}",
                    cache=True,
                ),
            ],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[RECORD_SYSTEM_FEEDBACK_TOOL],
            tool_choice={"type": "tool", "name": "record_system_feedback"},
            max_output_tokens=4096,
        )
        ctx = CallContext(
            session_id=session.id,
            task_id=task.id,
            agent="metareview",
            action="GenerateSystemFeedback",
            mode="system",
        )
        resp = await self.deps.llm.call(spec, ctx)
        record = self._final_tool_use(resp, "record_system_feedback")
        record_error = _system_feedback_record_error(record)
        if record_error:
            record = await self._recover_system_feedback(
                task=task,
                original_prompt=prompt,
                prior_record=record,
                failure_reason=record_error,
            )
        if record is None:
            return TaskResult(kind="noop", extra={"reason": "no record_system_feedback"})

        narrative = record.get("narrative") or ""
        if record.get("common_weaknesses"):
            narrative += "\n\n**Common weaknesses:** " + "; ".join(record["common_weaknesses"])
        if record.get("common_strengths"):
            narrative += "\n\n**Common strengths:** " + "; ".join(record["common_strengths"])
        if record.get("suggested_focus_areas"):
            narrative += "\n\n**Suggested focus:** " + "; ".join(record["suggested_focus_areas"])

        feedback_text = narrative.strip()
        if not feedback_text:
            previous = await fb_repo.latest_system_feedback(self.deps.db, session.id)
            if previous is not None and previous.text.strip():
                feedback_text = previous.text.strip()
                fallback_reason = "carried_forward_previous_feedback"
            else:
                feedback_text = (
                    "No substantive meta-review synthesis was returned at this "
                    "threshold. Continue prioritizing hypotheses that address "
                    "recurring reviewer critiques and recent tournament rationales."
                )
                fallback_reason = "empty_feedback_record"
            record = dict(record)
            record["narrative"] = feedback_text
            record["fallback_reason"] = fallback_reason
            log.warning("metareview_empty_feedback", fallback_reason=fallback_reason)

        fb_id = ids.feedback_id()
        artifact_path = await write_json(
            self.deps.cfg, session.id, "system_feedback", fb_id, record
        )
        await fb_repo.insert(
            self.deps.db,
            SystemFeedback(
                id=fb_id,
                session_id=session.id,
                created_at=datetime.now(UTC),
                source="meta_review",
                kind="system_feedback",
                target_id=None,
                text=feedback_text[:8000],
                artifact_path=artifact_path,
                active=True,
            ),
        )
        return TaskResult(
            kind="system_feedback_generated",
            extra={
                "feedback_id": fb_id,
                "n_reviews": len(reviews),
                "n_human_feedback": len(human_feedback),
            },
        )

    async def _recover_system_feedback(
        self,
        *,
        task: Task,
        original_prompt: str,
        prior_record: dict | None,
        failure_reason: str,
    ) -> dict | None:
        prompt = (
            "The previous meta-review response did not produce parseable structured "
            "feedback. Return no prose and call `record_system_feedback` exactly once. "
            "The payload must contain a non-empty `narrative` string plus arrays named "
            "`common_weaknesses`, `common_strengths`, and `suggested_focus_areas`. "
            "Recover the substantive scientific guidance from the prior payload without "
            "adding unsupported claims.\n\n"
            f"Validation failure: {failure_reason}.\n\n"
            f"# Previous invalid payload\n{str(prior_record)[:8000]}\n\n"
            f"# Original request tail\n{original_prompt[-6000:]}"
        )
        attempts = min(3, max(1, int(self.deps.cfg.tool_loop.metareview_max_iters)))
        last_record = prior_record
        for attempt in range(1, attempts + 1):
            spec = AgentCallSpec(
                route=route(self.deps.cfg, "metareview", "system"),
                system_blocks=[
                    CachedBlock(
                        "You are the meta-review agent. Return no prose. Call the "
                        "`record_system_feedback` tool with valid structured feedback.",
                        cache=False,
                    )
                ],
                user_blocks=[CachedBlock(prompt, cache=False)],
                tools=[RECORD_SYSTEM_FEEDBACK_TOOL],
                tool_choice={"type": "tool", "name": "record_system_feedback"},
                max_output_tokens=4096,
            )
            ctx = CallContext(
                session_id=task.session_id,
                task_id=task.id,
                agent="metareview",
                action="GenerateSystemFeedback",
                mode="system_recovery",
            )
            resp = await self.deps.llm.call(spec, ctx)
            last_record = self._final_tool_use(resp, "record_system_feedback")
            error = _system_feedback_record_error(last_record)
            if error is None:
                return last_record
            failure_reason = f"attempt {attempt}/{attempts}: {error}"
        log.warning("metareview_feedback_recovery_failed", err=failure_reason)
        return last_record

    @staticmethod
    def _human_feedback_block(feedback_rows: list[SystemFeedback]) -> str:
        if not feedback_rows:
            return ""

        lines = [
            "Active human feedback to incorporate into this meta-analysis:",
            "- directive/preference: treat as steering constraints for future proposals.",
            "- pin: treat as a hypothesis or direction the user wants explored further.",
            "- rejection: treat as a rejected hypothesis; extract only general lessons when the text gives a reason.",
        ]
        for f in feedback_rows[:50]:
            target = f.target_id or "global"
            feedback_text = " ".join((f.text or "").split())
            lines.append(f"- {f.kind} ({target}): {feedback_text[:1000]}")
        return "\n".join(lines)

    # ----------------------------- final overview ----------------------------- #

    async def _final_overview(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        top = await hyp_repo.top_by_elo(self.deps.db, session.id, k=10)
        all_hyps = await hyp_repo.list_for_session(self.deps.db, session.id)
        if not top and not all_hyps:
            return TaskResult(kind="noop", extra={"reason": "no hypotheses"})
        if not top:
            top = all_hyps[:10]

        # Fetch all reviews for the session in one query, then group by
        # hypothesis_id. Beats N+1 list_for_hypothesis() calls for top-K.
        reviews_by_hyp: dict[str, list] = {}
        for rv in await rev_repo.list_for_session(self.deps.db, session.id):
            reviews_by_hyp.setdefault(rv.hypothesis_id, []).append(rv)

        # Build the top-hypotheses block: summary + best review + winning rationale
        chunks: list[str] = []
        for h in top:
            review_lines: list[str] = []
            for r in reviews_by_hyp.get(h.id, []):
                review_lines.append(
                    f"  - {r.kind}: verdict={r.verdict or '?'} "
                    f"(n={r.scores.novelty}, c={r.scores.correctness}, t={r.scores.testability})"
                )
            elo_s = f"{h.elo:.0f}" if h.elo is not None else "—"
            chunks.append(
                f"### `{h.id}` (Elo {elo_s}, strategy `{h.strategy}`)\n"
                f"**Title.** {h.title}\n\n"
                f"**Hypothesis and study plan.**\n\n"
                f"{_hypothesis_overview_context(h.full_text)}\n\n"
                f"**Reviews:**\n" + ("\n".join(review_lines) or "  (none)")
            )
        top_block = "\n\n---\n\n".join(chunks)

        latest_fb = await fb_repo.latest_system_feedback(self.deps.db, session.id)

        prompt = render(
            "metareview.final",
            goal=session.research_plan.objective,
            preferences="; ".join(session.research_plan.preferences),
            system_feedback=latest_fb.text if latest_fb else "",
            top_hypotheses_block=top_block,
        )
        r = route(self.deps.cfg, "metareview", "final")
        spec = AgentCallSpec(
            route=r,
            system_blocks=[
                CachedBlock(self._system_prompt_header(), cache=True),
                CachedBlock(
                    f"# Research goal\n{session.research_goal}\n\n"
                    f"# Preferences\n{'; '.join(session.research_plan.preferences)}",
                    cache=True,
                ),
            ],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[],  # No tools — write the markdown directly
            tool_choice=None,
            max_output_tokens=8192,
        )
        ctx = CallContext(
            session_id=session.id,
            task_id=task.id,
            agent="metareview",
            action="GenerateFinalResearchOverview",
            mode="final",
        )
        resp = await self.deps.llm.call(spec, ctx)
        text = self._final_text(resp)
        if not text.strip():
            text = "# Research overview\n\n_(No content was generated; see transcripts.)_"
        citation_appendix = render_hypothesis_citation_appendix(top)
        if citation_appendix:
            text = text.rstrip() + "\n\n" + citation_appendix + "\n"

        overview_path = await write_text(
            self.deps.cfg, session.id, "final", "overview", ".md", text
        )
        return TaskResult(
            kind="final_overview_generated",
            extra={"overview_path": overview_path, "n_top": len(top)},
        )


def _system_feedback_record_error(record: dict | None) -> str | None:
    if record is None:
        return "missing record_system_feedback tool call"
    if record.get("_raw_arguments") is not None:
        return "record_system_feedback arguments were unparseable or truncated"
    if not isinstance(record.get("narrative"), str):
        return "missing or invalid narrative"
    for field in (
        "common_weaknesses",
        "common_strengths",
        "suggested_focus_areas",
    ):
        values = record.get(field)
        if values is not None and (
            not isinstance(values, list) or any(not isinstance(value, str) for value in values)
        ):
            return f"missing or invalid {field} array"
    return None


def _hypothesis_overview_context(text: str, *, max_chars: int = 6000) -> str:
    """Keep the study-plan lead and grounding report inside final-review context."""
    if len(text) <= max_chars:
        return text
    marker = "## Capability grounding"
    marker_index = text.find(marker)
    if marker_index < 0:
        return text[:max_chars]
    grounding = text[marker_index : marker_index + min(2000, max_chars // 3)]
    prefix_budget = max(1000, max_chars - len(grounding) - 40)
    return f"{text[:prefix_budget]}\n\n[...middle omitted...]\n\n{grounding}"
