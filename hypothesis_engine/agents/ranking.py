# Modified from the original work.
"""Ranking agent — manages the Elo tournament.

Two actions:
- `AddToTournament(hypothesis_id)` — initialize Elo + state. No LLM call.
- `RunTournamentBatch(focus_id?)` — pick a pair, debate, parse verdict, apply Elo.

Pair selection mixes new-arrival pairings, similar-Elo/proximity-near pairs,
and an occasional random pull.
Debate mode is preferred when matches are new or Elo gap is small; pairwise
otherwise.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from .. import ids
from ..llm.anthropic_client import AgentCallSpec, CachedBlock, CallContext
from ..llm.prompts import render
from ..llm.routing import route
from ..logging import get_logger
from ..models import Hypothesis, Task, TaskResult, TournamentMatch
from ..safety.quoting import quote_hypothesis
from ..storage.repos import (
    hypotheses as hyp_repo,
)
from ..storage.repos import (
    reviews as rev_repo,
)
from ..storage.repos import (
    sessions as sess_repo,
)
from ..storage.repos import (
    tournaments as tourney_repo,
)
from ..vectors.embedder import make_embedder
from ..vectors.store import FaissStore
from .base import AgentDeps, BaseAgent

log = get_logger("ranking")


PairMode = Literal["pairwise", "debate"]
PromptSideName = Literal["a", "b"]


@dataclass(frozen=True)
class RankingPromptSide:
    side: PromptSideName
    hypothesis: Hypothesis
    hypothesis_text: str
    review_text: str

    @property
    def total_chars(self) -> int:
        return len(self.hypothesis_text) + len(self.review_text)


@dataclass(frozen=True)
class RankingPromptLayout:
    prompt1: RankingPromptSide
    prompt2: RankingPromptSide
    order_key: str

    def winner_side_for_choice(self, choice: int) -> PromptSideName:
        return self.prompt1.side if choice == 1 else self.prompt2.side

    def prompt_position_for_side(self, side: str) -> int | None:
        if side == self.prompt1.side:
            return 1
        if side == self.prompt2.side:
            return 2
        return None


class RankingAgent(BaseAgent):
    name = "ranking"

    def __init__(self, deps: AgentDeps) -> None:
        super().__init__(deps)

    async def execute(self, task: Task) -> TaskResult:
        if task.action == "AddToTournament":
            return await self._add_to_tournament(task)
        if task.action == "RunTournamentBatch":
            return await self._run_tournament_batch(task)
        raise ValueError(f"RankingAgent does not handle action {task.action!r}")

    # ----------------------------- AddToTournament ----------------------------- #

    async def _add_to_tournament(self, task: Task) -> TaskResult:
        hypothesis_id = task.target_id
        if not hypothesis_id:
            raise ValueError("AddToTournament requires target_id")
        changed = await hyp_repo.init_tournament(
            self.deps.db, hypothesis_id,
            initial_elo=float(self.deps.cfg.ranking.elo_initial),
        )
        return TaskResult(
            kind="added_to_tournament",
            hypothesis_ids=[hypothesis_id] if changed else [],
            extra={"already_in_tournament": not changed},
        )

    # ----------------------------- RunTournamentBatch -------------------------- #

    async def _run_tournament_batch(self, task: Task) -> TaskResult:
        session = await sess_repo.fetch(self.deps.db, task.session_id)
        if session is None:
            raise RuntimeError(f"session {task.session_id} missing")

        candidates = await hyp_repo.tournament_candidates(self.deps.db, session.id)
        if len(candidates) < 2:
            return TaskResult(kind="noop", extra={"reason": "fewer than 2 candidates"})

        focus_id = task.payload.get("focus")
        reserved = await self._select_and_reserve_pair(
            session.id, candidates, focus_id=focus_id, task_id=task.id
        )
        if reserved is None:
            return TaskResult(kind="noop", extra={"reason": "no pair available"})
        hyp_a, hyp_b, similarity, pair_key = reserved

        try:
            return await self._run_reserved_match(session, task, hyp_a, hyp_b, similarity)
        finally:
            try:
                await tourney_repo.release_pair(
                    self.deps.db, session_id=session.id, pair_key=pair_key, task_id=task.id
                )
            except Exception as exc:
                log.warning(
                    "pair_release_failed",
                    session_id=session.id,
                    pair_key=pair_key,
                    task_id=task.id,
                    err=str(exc)[:200],
                )

    async def _run_reserved_match(
        self,
        session,
        task: Task,
        hyp_a: Hypothesis,
        hyp_b: Hypothesis,
        similarity: float | None,
    ) -> TaskResult:
        mode = self._select_mode(hyp_a, hyp_b)
        verdict, rationale, transcript_id, failure_reason, prompt_layout = await self._run_debate(
            session, hyp_a, hyp_b, mode=mode, round_id=task.id
        )
        # Derive the round_id deterministically from the task id so that a
        # crash-then-retry computes the *same* match_id. `apply_elo_update`
        # below is idempotent on `match_id` — a non-deterministic round_id
        # (e.g. a wall-clock timestamp) defeats that and would double-apply
        # the Elo delta on retry.
        round_id = task.id
        if verdict is None:
            # Parsing failed — record an invalid match and don't update Elo.
            mid_invalid = ids.match_id(hyp_a.id, hyp_b.id, round_id)
            await tourney_repo.insert_match(self.deps.db, TournamentMatch(
                id=mid_invalid, session_id=session.id,
                created_at=datetime.now(UTC),
                hyp_a=hyp_a.id, hyp_b=hyp_b.id, mode="invalid", winner=None,
                elo_a_before=hyp_a.elo or 1200.0, elo_b_before=hyp_b.elo or 1200.0,
                rationale=rationale, transcript_id=transcript_id, similarity=similarity,
                **_prompt_match_fields(prompt_layout, winner=None),
            ))
            reason = failure_reason or "unparseable verdict"
            log.warning("ranking_invalid_verdict", a=hyp_a.id, b=hyp_b.id, reason=reason)
            return TaskResult(
                kind="noop",
                extra={
                    "reason": reason,
                    "mode": mode,
                    "hyp_a": hyp_a.id,
                    "hyp_b": hyp_b.id,
                    "transcript_id": transcript_id,
                },
            )

        elo_a_before = float(hyp_a.elo or self.deps.cfg.ranking.elo_initial)
        elo_b_before = float(hyp_b.elo or self.deps.cfg.ranking.elo_initial)
        mid = ids.match_id(hyp_a.id, hyp_b.id, round_id)
        await tourney_repo.insert_match(self.deps.db, TournamentMatch(
            id=mid, session_id=session.id,
            created_at=datetime.now(UTC),
            hyp_a=hyp_a.id, hyp_b=hyp_b.id, mode=mode, winner=verdict,
            elo_a_before=elo_a_before, elo_b_before=elo_b_before,
            elo_a_after=None, elo_b_after=None,
            rationale=rationale, transcript_id=transcript_id, similarity=similarity,
            **_prompt_match_fields(prompt_layout, winner=verdict),
        ))

        pair_status = await tourney_repo.pair_status(
            self.deps.db,
            session.id,
            hyp_a.id,
            hyp_b.id,
            max_pair_matches=int(self.deps.cfg.ranking.pair_max_matches),
            wins_to_close_pair=int(self.deps.cfg.ranking.pair_wins_to_close),
        )
        pair_resolved = bool(pair_status.get("closed"))
        pair_winner_hyp_id = pair_status.get("winner_hyp_id")
        aggregate_winner = None
        if pair_winner_hyp_id == hyp_a.id:
            aggregate_winner = "a"
        elif pair_winner_hyp_id == hyp_b.id:
            aggregate_winner = "b"

        applied = False
        if pair_resolved and aggregate_winner is not None:
            applied = await tourney_repo.apply_elo_update(
                self.deps.db,
                match_id=mid, hyp_a=hyp_a.id, hyp_b=hyp_b.id, winner=aggregate_winner,
                elo_a_before=elo_a_before, elo_b_before=elo_b_before,
                k_new=self.deps.cfg.ranking.k_factor_new,
                k_warm=self.deps.cfg.ranking.k_factor_warm,
                logistic_scale=self.deps.cfg.ranking.elo_logistic_scale,
                elo_initial=self.deps.cfg.ranking.elo_initial,
            )
        log.info(
            "match_complete",
            mode=mode, hyp_a=hyp_a.id, hyp_b=hyp_b.id, winner=verdict,
            pair_resolved=pair_resolved, pair_winner=pair_winner_hyp_id,
            pair_round=pair_status.get("valid_matches"),
            elo_applied=applied, similarity=similarity,
        )
        return TaskResult(
            kind="tournament_match_complete",
            match_ids=[mid],
            hypothesis_ids=[hyp_a.id, hyp_b.id],
            extra={
                "mode": mode,
                "winner": verdict,
                "pair_resolved": pair_resolved,
                "pair_winner": pair_winner_hyp_id,
                "pair_round": pair_status.get("valid_matches"),
                "elo_applied": applied,
            },
        )

    # ----------------------------- pair selection ----------------------------- #

    async def _select_pair(
        self,
        session_id: str,
        candidates: list[Hypothesis],
        *,
        focus_id: str | None,
        blocked_pair_keys: set[str] | None = None,
    ) -> tuple[Hypothesis, Hypothesis, float | None] | None:
        # Build the FAISS store once for this pair selection — every prior
        # iteration re-instantiated the embedder, re-read index.faiss + JSON
        # off disk, and reconstructed the entire index just to dot-product two
        # rows. With ~20 pair candidates per RunTournamentBatch that was
        # ~20 full-index reloads and reconstructions for a single match.
        store = await self._load_store(session_id)
        blocked_pair_keys = blocked_pair_keys or set()

        if focus_id:
            focus = next((h for h in candidates if h.id == focus_id), None)
            if focus is not None:
                pool = self._eligible_pool(
                    focus, [h for h in candidates if h.id != focus_id], blocked_pair_keys
                )
                opp = self._nearest_elo(focus, pool, store=store)
                if opp is not None:
                    sim = self._similarity(store, focus, opp)
                    return focus, opp, sim

        new_hyps = [h for h in candidates if h.matches_played < 3]
        warm = [h for h in candidates if h.matches_played >= 3]

        warmup_pair = self._select_warmup_pair(
            store, new_hyps, warm, blocked_pair_keys=blocked_pair_keys
        )
        if warmup_pair is not None:
            return warmup_pair

        cfg = self.deps.cfg.ranking
        r = random.random()
        # Bucket 1: pair a new hypothesis with nearest-Elo warm/stable.
        if r < cfg.p_new and new_hyps and warm:
            for a in random.sample(new_hyps, len(new_hyps)):
                pool = self._eligible_pool(a, warm, blocked_pair_keys)
                b = self._nearest_elo(a, pool, store=store)
                if b is not None:
                    return a, b, self._similarity(store, a, b)

        # Bucket 2: similar-Elo pair within the warm set, weighted toward
        # proximity-near ideas, matching the paper's tournament description.
        if r < cfg.p_new + cfg.p_close and len(warm) >= 2:
            pair = self._sample_close_elo(
                store, warm, blocked_pair_keys=blocked_pair_keys
            )
            if pair is not None:
                return pair

        # Bucket 3: random Elo-weighted (top-heavy)
        if len(candidates) >= 2:
            sorted_by_elo = sorted(candidates, key=lambda h: -(h.elo or 1200))
            top = sorted_by_elo[: max(2, len(candidates) // 2)]
            if len(top) >= 2:
                pairs = [
                    (a, b)
                    for i, a in enumerate(top)
                    for b in top[i + 1:]
                    if _pair_key(a.id, b.id) not in blocked_pair_keys
                ]
                if not pairs:
                    return None
                a, b = random.choice(pairs)
                return a, b, self._similarity(store, a, b)
        return None

    async def _select_and_reserve_pair(
        self,
        session_id: str,
        candidates: list[Hypothesis],
        *,
        focus_id: str | None,
        task_id: str,
    ) -> tuple[Hypothesis, Hypothesis, float | None, str] | None:
        active_pair_keys = await tourney_repo.active_pair_keys(self.deps.db, session_id)
        closed_pair_keys = await tourney_repo.closed_pair_keys(
            self.deps.db,
            session_id,
            max_pair_matches=int(self.deps.cfg.ranking.pair_max_matches),
            wins_to_close_pair=int(self.deps.cfg.ranking.pair_wins_to_close),
        )
        blocked_pair_keys = active_pair_keys | closed_pair_keys
        max_attempts = max(1, len(candidates) * (len(candidates) - 1) // 2)
        for _attempt in range(max_attempts):
            pair = await self._select_pair(
                session_id,
                candidates,
                focus_id=focus_id,
                blocked_pair_keys=blocked_pair_keys,
            )
            if pair is None:
                return None
            hyp_a, hyp_b, similarity = pair
            pair_key = _pair_key(hyp_a.id, hyp_b.id)
            if await tourney_repo.reserve_pair(
                self.deps.db,
                session_id=session_id,
                pair_key=pair_key,
                task_id=task_id,
                max_pair_matches=int(self.deps.cfg.ranking.pair_max_matches),
                wins_to_close_pair=int(self.deps.cfg.ranking.pair_wins_to_close),
            ):
                oriented_a_id, oriented_b_id = await tourney_repo.next_pair_orientation(
                    self.deps.db, session_id, hyp_a.id, hyp_b.id
                )
                by_id = {hyp_a.id: hyp_a, hyp_b.id: hyp_b}
                return by_id[oriented_a_id], by_id[oriented_b_id], similarity, pair_key
            blocked_pair_keys.add(pair_key)
        return None

    def _select_warmup_pair(
        self,
        store: FaissStore | None,
        new_hyps: list[Hypothesis],
        warm: list[Hypothesis],
        *,
        blocked_pair_keys: set[str] | None = None,
    ) -> tuple[Hypothesis, Hypothesis, float | None] | None:
        """Prioritize maturing the pool before idle evolution can start."""
        blocked_pair_keys = blocked_pair_keys or set()
        min_mature = max(0, int(self.deps.cfg.evolution.min_mature))
        if not new_hyps or len(warm) >= min_mature:
            return None

        challengers = sorted(
            new_hyps,
            key=lambda h: (h.matches_played, h.elo or 1200.0),
            reverse=True,
        )
        for challenger in challengers:
            if warm:
                pool = self._eligible_pool(challenger, warm, blocked_pair_keys)
            else:
                pool = self._eligible_pool(
                    challenger,
                    [h for h in new_hyps if h.id != challenger.id],
                    blocked_pair_keys,
                )
            opponent = self._nearest_elo(challenger, pool, store=store)
            if opponent is not None:
                return challenger, opponent, self._similarity(store, challenger, opponent)
        return None

    def _eligible_pool(
        self,
        target: Hypothesis,
        pool: list[Hypothesis],
        blocked_pair_keys: set[str],
    ) -> list[Hypothesis]:
        return [
            h for h in pool
            if _pair_key(target.id, h.id) not in blocked_pair_keys
        ]

    def _nearest_elo(
        self,
        target: Hypothesis,
        pool: list[Hypothesis],
        *,
        store: FaissStore | None = None,
    ) -> Hypothesis | None:
        if not pool:
            return None
        target_elo = target.elo or self.deps.cfg.ranking.elo_initial
        deltas = [abs((h.elo or self.deps.cfg.ranking.elo_initial) - target_elo) for h in pool]
        best_delta = min(deltas)
        balance_window = max(
            float(self.deps.cfg.ranking.k_factor_new),
            float(self.deps.cfg.ranking.k_factor_warm),
            16.0,
        )
        comparable = [
            h for h, delta in zip(pool, deltas, strict=True)
            if delta <= best_delta + balance_window
        ]

        def _similarity_score(h: Hypothesis) -> float:
            sim = self._similarity(store, target, h)
            return sim if sim is not None else 0.0

        return min(
            comparable,
            key=lambda h: (
                h.matches_played,
                -_similarity_score(h),
                abs((h.elo or self.deps.cfg.ranking.elo_initial) - target_elo),
                h.id,
            ),
        )

    def _sample_close_elo(
        self,
        store: FaissStore | None,
        pool: list[Hypothesis],
        *,
        blocked_pair_keys: set[str] | None = None,
    ) -> tuple[Hypothesis, Hypothesis, float | None] | None:
        """Among pairs with |Δelo|<200, sample by Elo closeness and proximity."""
        if len(pool) < 2:
            return None
        blocked_pair_keys = blocked_pair_keys or set()
        # Build a small candidate list of pairs (cap to keep cost low)
        weights: list[float] = []
        pairs: list[tuple[Hypothesis, Hypothesis, float | None]] = []
        for i, a in enumerate(pool):
            for b in pool[i + 1:]:
                if _pair_key(a.id, b.id) in blocked_pair_keys:
                    continue
                d_elo = abs((a.elo or 1200) - (b.elo or 1200))
                if d_elo > 200:
                    continue
                sim = self._similarity(store, a, b)
                similarity_score = (sim + 1.0) / 2.0 if sim is not None else 0.5
                w = float(np.exp(-d_elo / 200.0)) * max(similarity_score, 0.05)
                weights.append(w)
                pairs.append((a, b, sim))
                if len(pairs) >= 20:  # cap
                    break
            if len(pairs) >= 20:
                break
        if not pairs:
            return None
        total = sum(weights)
        if total <= 0:
            return random.choice(pairs)
        r = random.uniform(0, total)
        cum = 0.0
        for w, pair in zip(weights, pairs, strict=True):
            cum += w
            if cum >= r:
                return pair
        return pairs[-1]

    async def _load_store(self, session_id: str) -> FaissStore | None:
        """Instantiate + load the session FAISS store once for pair selection."""
        try:
            embedder = make_embedder(self.deps.cfg)
        except (RuntimeError, ValueError):
            return None
        store = FaissStore(self.deps.cfg, session_id, dim=embedder.dim)
        await store.load_or_create()
        if store.n == 0:
            return None
        return store

    def _similarity(
        self, store: FaissStore | None, a: Hypothesis, b: Hypothesis
    ) -> float | None:
        """Cosine via the session's FAISS store (already L2-normalized).

        Reconstructs only the two rows we need (O(2·dim)) — the previous
        version called `reconstruct_n(0, n)` for every pair, materialising
        the full N×dim matrix just to read two rows.
        """
        if store is None or store.index is None or store.n == 0:
            return None
        i = store.offset_of(a.id)
        j = store.offset_of(b.id)
        if i is None or j is None:
            return None
        vec_i = store.index.reconstruct(int(i))
        vec_j = store.index.reconstruct(int(j))
        return float(vec_i @ vec_j)

    # ----------------------------- mode selection ----------------------------- #

    def _select_mode(self, a: Hypothesis, b: Hypothesis) -> PairMode:
        cfg = self.deps.cfg.ranking
        if min(a.matches_played, b.matches_played) < cfg.debate_when_matches_lt:
            return "debate"
        if abs((a.elo or 1200) - (b.elo or 1200)) < cfg.debate_when_elo_delta_lt:
            return "debate"
        return "pairwise"

    # ----------------------------- the debate / pairwise call ----------------- #

    async def _run_debate(
        self,
        session,
        a: Hypothesis,
        b: Hypothesis,
        *,
        mode: PairMode,
        round_id: str,
    ) -> tuple[Literal["a", "b"] | None, str, str | None, str | None, RankingPromptLayout]:
        plan = session.research_plan
        review_a = await self._best_review(a.id)
        review_b = await self._best_review(b.id)
        layout = self._build_prompt_layout(session.id, round_id, a, b, review_a, review_b)

        template = "ranking.debate" if mode == "debate" else "ranking.pairwise"
        prompt_kwargs = {
            "goal": plan.objective,
            "preferences": "; ".join(plan.preferences),
            "idea_attributes": "; ".join(plan.idea_attributes),
            "hypothesis_1_id": layout.prompt1.hypothesis.id,
            "hypothesis_1": quote_hypothesis(layout.prompt1.hypothesis_text, id_=layout.prompt1.hypothesis.id),
            "hypothesis_2_id": layout.prompt2.hypothesis.id,
            "hypothesis_2": quote_hypothesis(layout.prompt2.hypothesis_text, id_=layout.prompt2.hypothesis.id),
            "review_1": layout.prompt1.review_text or "(no review)",
            "review_2": layout.prompt2.review_text or "(no review)",
            "notes": (
                "Use the same criteria and roughly equal attention for both hypotheses. "
                "Do not reward a hypothesis merely because it has more text. Treat missing detail as a weakness "
                "only when the missing scientific content matters. End your response with the line: "
                "better idea: <1 or 2>"
            ),
        }
        prompt = render(template, **prompt_kwargs)

        r = route(self.deps.cfg, "ranking", "debate" if mode == "debate" else "pairwise")

        system = [
            CachedBlock(self._system_prompt_header(), cache=True),
            CachedBlock(
                f"# Research goal\n{session.research_goal}\n\n"
                f"# Preferences\n{'; '.join(plan.preferences)}\n\n"
                "Conclude every response with the exact line `better idea: 1` or "
                "`better idea: 2`. No other format. Do not call any tools.",
                cache=True,
            ),
        ]
        max_output_tokens = (
            self.deps.cfg.ranking.debate_max_output_tokens
            if mode == "debate"
            else self.deps.cfg.ranking.pairwise_max_output_tokens
        )
        spec = AgentCallSpec(
            route=r,
            system_blocks=system,
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[],
            tool_choice=None,
            max_output_tokens=max_output_tokens,
            stop_sequences=None,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="ranking", action="RunTournamentBatch", mode=mode,
        )
        resp = await self.deps.llm.call(spec, ctx)
        text = self._final_text(resp)
        choice = _parse_better_idea(text)
        transcript_id = resp.transcript_id
        failure_reason: str | None = None
        if choice is None:
            stop_reason = _stop_reason(resp)
            failure_reason = (
                "max_tokens without verdict"
                if stop_reason == "max_tokens"
                else "missing better idea verdict"
            )
            retry_choice, retry_text, retry_transcript_id = await self._retry_verdict_only(
                session=session,
                prompt1=layout.prompt1,
                prompt2=layout.prompt2,
                mode=mode,
                prior_text=text,
                stop_reason=stop_reason,
            )
            if retry_choice is not None:
                choice = retry_choice
                transcript_id = retry_transcript_id or transcript_id
                text = (
                    text
                    + "\n\n# Verdict recovery\n"
                    + retry_text
                ).strip()
                failure_reason = None

        if choice is None:
            return None, text, transcript_id, failure_reason, layout

        winner = layout.winner_side_for_choice(choice)
        return winner, text, transcript_id, None, layout

    async def _retry_verdict_only(
        self,
        *,
        session,
        prompt1: RankingPromptSide,
        prompt2: RankingPromptSide,
        mode: PairMode,
        prior_text: str,
        stop_reason: str | None,
    ) -> tuple[int | None, str, str | None]:
        prior_tail = prior_text[-3000:] if prior_text else "(no usable prior response text)"
        prompt = (
            "The previous ranking response did not produce a parseable final verdict"
            f" (stop_reason={stop_reason or 'unknown'}). Do not continue the debate.\n\n"
            "Choose the stronger hypothesis for the research goal below and answer in at most "
            "five sentences. The final line must be exactly `better idea: 1` or `better idea: 2`.\n\n"
            f"# Research goal\n{session.research_goal}\n\n"
            f"# Hypothesis 1 ({prompt1.hypothesis.id})\n"
            f"{quote_hypothesis(prompt1.hypothesis_text, id_=prompt1.hypothesis.id)}\n\n"
            f"# Review 1\n{prompt1.review_text or '(no review)'}\n\n"
            f"# Hypothesis 2 ({prompt2.hypothesis.id})\n"
            f"{quote_hypothesis(prompt2.hypothesis_text, id_=prompt2.hypothesis.id)}\n\n"
            f"# Review 2\n{prompt2.review_text or '(no review)'}\n\n"
            f"# Previous partial response tail\n{prior_tail}\n"
        )
        r = route(self.deps.cfg, "ranking", "debate" if mode == "debate" else "pairwise")
        spec = AgentCallSpec(
            route=r,
            system_blocks=[
                CachedBlock(
                    "You are the ranking agent. Return a compact final judgment only. "
                    "End with exactly `better idea: 1` or `better idea: 2`.",
                    cache=False,
                )
            ],
            user_blocks=[CachedBlock(prompt, cache=False)],
            tools=[],
            tool_choice=None,
            max_output_tokens=self.deps.cfg.ranking.verdict_retry_max_output_tokens,
            stop_sequences=None,
        )
        ctx = CallContext(
            session_id=session.id, task_id=None,
            agent="ranking", action="RunTournamentBatch", mode=f"{mode}_verdict_retry",
        )
        resp = await self.deps.llm.call(spec, ctx)
        text = self._final_text(resp)
        return _parse_better_idea(text), text, resp.transcript_id


    def _build_prompt_layout(
        self,
        session_id: str,
        round_id: str,
        a: Hypothesis,
        b: Hypothesis,
        review_a: str | None,
        review_b: str | None,
    ) -> RankingPromptLayout:
        side_a = self._prompt_side("a", a, review_a)
        side_b = self._prompt_side("b", b, review_b)
        order_key = _prompt_order_key(session_id, round_id, a.id, b.id)
        return RankingPromptLayout(prompt1=side_a, prompt2=side_b, order_key=order_key)

    def _prompt_side(
        self, side: PromptSideName, hypothesis: Hypothesis, review: str | None
    ) -> RankingPromptSide:
        cfg = self.deps.cfg.ranking
        hyp_text = _ranking_hypothesis_text(
            hypothesis, max_chars=int(cfg.prompt_hypothesis_max_chars)
        )
        review_text = _clip_for_ranking_prompt(
            review or "", int(cfg.prompt_review_max_chars)
        )
        total_max = int(cfg.prompt_side_max_chars)
        if total_max > 0 and len(hyp_text) + len(review_text) > total_max:
            review_budget = max(0, total_max - len(hyp_text))
            review_text = _clip_for_ranking_prompt(review_text, review_budget)
            if len(hyp_text) + len(review_text) > total_max:
                hyp_text = _clip_for_ranking_prompt(hyp_text, total_max)
                review_text = ""
        return RankingPromptSide(
            side=side, hypothesis=hypothesis, hypothesis_text=hyp_text, review_text=review_text
        )

    async def _best_review(self, hypothesis_id: str) -> str | None:
        rs = await rev_repo.list_for_hypothesis(self.deps.db, hypothesis_id)
        if not rs:
            return None
        # Prefer 'full' kind if present.
        rs_sorted = sorted(rs, key=lambda r: (r.kind != "full", -(r.scores.novelty or 0)))
        return rs_sorted[0].body


def _prompt_order_key(session_id: str, round_id: str, a_id: str, b_id: str) -> str:
    payload = f"{session_id}:{round_id}:{a_id}:{b_id}:ranking-prompt-order"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ranking_hypothesis_text(hypothesis: Hypothesis, *, max_chars: int) -> str:
    parts = [
        f"Title: {hypothesis.title}",
        f"Summary: {hypothesis.summary}",
        "Full hypothesis:",
        hypothesis.full_text or "",
    ]
    return _clip_for_ranking_prompt("\n".join(part for part in parts if part), max_chars)


def _clip_for_ranking_prompt(text: str, max_chars: int) -> str:
    cleaned = "\n".join(line.rstrip() for line in str(text or "").splitlines()).strip()
    if max_chars < 0:
        return cleaned
    if max_chars == 0:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    marker = "\n[truncated to balanced ranking prompt budget]"
    if max_chars <= len(marker):
        return cleaned[:max_chars].rstrip()
    keep = max_chars - len(marker)
    return cleaned[:keep].rstrip() + marker

def _prompt_match_fields(layout: RankingPromptLayout, *, winner: str | None) -> dict[str, object]:
    return {
        "prompt1_hyp_id": layout.prompt1.hypothesis.id,
        "prompt2_hyp_id": layout.prompt2.hypothesis.id,
        "prompt1_side": layout.prompt1.side,
        "prompt2_side": layout.prompt2.side,
        "winner_prompt_position": layout.prompt_position_for_side(winner) if winner else None,
        "prompt1_chars": layout.prompt1.total_chars,
        "prompt2_chars": layout.prompt2.total_chars,
        "prompt_order_key": layout.order_key,
    }


def _stop_reason(response) -> str | None:
    return getattr(response.raw, "stop_reason", None)


_VERDICT_DIGIT_RE = re.compile(r"^[\W_]*\**\s*([12])\b")


def _pair_key(a_id: str, b_id: str) -> str:
    a, b = sorted((a_id, b_id))
    return f"{a}::{b}"


def _parse_better_idea(text: str) -> int | None:
    """Find the trailing 'better idea: 1|2' marker (case-insensitive, any line).

    The previous implementation used `"1" in tail.split()[0:1]`, which is
    `True` only when the first whitespace-token *equals* "1" exactly. That
    rejected valid replies like 'better idea: option 1' or 'better idea: **1
    because...'. The regex anchors at the start and matches the first 1 or 2
    as a word boundary so we accept all those forms while still rejecting
    'better idea: 12' (which the boundary check excludes).
    """
    if not text:
        return None
    lines = text.strip().splitlines()
    for line in reversed(lines):
        low = line.strip().lower()
        if "better idea" in low and ":" in low:
            tail = low.split(":", 1)[1].strip()
            m = _VERDICT_DIGIT_RE.match(tail)
            if m:
                return int(m.group(1))
            # Common phrasing: "option 1", "hypothesis 1", "hyp 1"
            for keyword in ("option", "hypothesis", "hyp"):
                if tail.startswith(keyword):
                    rest = tail[len(keyword):].lstrip()
                    m2 = _VERDICT_DIGIT_RE.match(rest)
                    if m2:
                        return int(m2.group(1))
    return None
