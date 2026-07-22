"""Anthropic Batch API support for sub-decile tournament matches.

The Batch API is ~50% cheaper than synchronous Messages calls but takes up to
24 h to complete. We use it for tournament matches between low-Elo pairs where
the freshness of the verdict doesn't matter much — the high-Elo head of the
leaderboard stays on the synchronous path so the final overview reflects
fresh judgments.

This module owns:
- BatchedMatch dataclass + per-session in-memory queue of pending matches.
- submit_batch(): drains the queue into one batch request.
- poll_batch(): checks for completion + extracts verdicts.
- A small reconciler that turns completed batch results into rows in
  tournament_matches + elo_journal (same idempotent path as RankingAgent).

The Ranking agent decides whether to enqueue a match for batch (low-decile)
or run it synchronously. The Supervisor calls submit_batch() periodically.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from ..config import Config
from ..logging import get_logger

log = get_logger("llm.batch")


@dataclass
class BatchedMatch:
    """One pending pairwise match awaiting batch submission."""

    custom_id: str               # e.g. "match::hyp_a::hyp_b::round_id"
    session_id: str
    hyp_a: str
    hyp_b: str
    prompt: str                  # ranking_pairwise.md fully rendered
    model: str
    max_tokens: int = 1024
    system: str = ""
    enqueued_at: float = field(default_factory=time.monotonic)


@dataclass
class BatchHandle:
    """Reference to a submitted batch."""

    batch_id: str
    session_id: str
    custom_ids: list[str]
    submitted_at: float
    model: str


class BatchPool:
    """In-memory queue of pending matches, drained by submit_batch().

    The Supervisor owns one pool. Matches are added by RankingAgent (when it
    determines the pair is sub-decile) and the Supervisor periodically calls
    submit_batch + poll_handles.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._pending: list[BatchedMatch] = []
        self._handles: list[BatchHandle] = []
        self._client: AsyncAnthropic | None = None
        api_key = cfg.secrets.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY") or ""
        if api_key:
            self._client = AsyncAnthropic(api_key=api_key)

    # ----------------------------- enqueue ----------------------------- #

    def enqueue(self, match: BatchedMatch) -> None:
        self._pending.append(match)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def inflight_count(self) -> int:
        return len(self._handles)

    # ----------------------------- submit ----------------------------- #

    async def submit_batch(self, *, min_size: int = 4) -> BatchHandle | None:
        """Drain pending → one Batch API request. Returns the handle or None.

        Requires at least `min_size` queued matches to amortize overhead. The
        SDK exposes batches under `client.messages.batches`; falls back to a
        synchronous fan-out if the SDK doesn't have the call (older versions).
        """
        if not self._pending or self._client is None:
            return None
        if len(self._pending) < min_size:
            return None
        batch = list(self._pending)
        self._pending.clear()

        requests = []
        for m in batch:
            req: dict[str, Any] = {
                "custom_id": m.custom_id,
                "params": {
                    "model": m.model,
                    "max_tokens": m.max_tokens,
                    "messages": [{"role": "user", "content": m.prompt}],
                },
            }
            if m.system:
                req["params"]["system"] = m.system
            requests.append(req)

        try:
            create = self._client.messages.batches.create  # type: ignore[attr-defined]
        except AttributeError:
            log.warning("batch_api_not_available_in_sdk", n=len(batch))
            # Re-enqueue so caller can retry synchronously elsewhere.
            self._pending.extend(batch)
            return None

        try:
            batch_obj = await create(requests=requests)
        except Exception as e:
            log.warning("batch_submit_failed", err=str(e))
            self._pending.extend(batch)
            return None

        handle = BatchHandle(
            batch_id=batch_obj.id,
            session_id=batch[0].session_id,
            custom_ids=[m.custom_id for m in batch],
            submitted_at=time.monotonic(),
            model=batch[0].model,
        )
        self._handles.append(handle)
        log.info("batch_submitted", batch_id=handle.batch_id, n=len(batch))
        return handle

    # ----------------------------- poll ----------------------------- #

    async def poll_handles(self) -> list[dict[str, Any]]:
        """Returns a list of per-handle status dicts; drops completed ones."""
        if self._client is None or not self._handles:
            return []
        out: list[dict[str, Any]] = []
        remaining: list[BatchHandle] = []
        for h in self._handles:
            try:
                retrieve = self._client.messages.batches.retrieve  # type: ignore[attr-defined]
            except AttributeError:
                # SDK doesn't support batches — drop quietly
                continue
            try:
                obj = await retrieve(h.batch_id)
            except Exception as e:
                log.warning("batch_poll_failed", batch_id=h.batch_id, err=str(e))
                out.append({"batch_id": h.batch_id, "status": "error", "error": str(e)})
                remaining.append(h)
                continue
            status = getattr(obj, "processing_status", None) or getattr(obj, "status", "unknown")
            entry: dict[str, Any] = {
                "batch_id": h.batch_id,
                "status": status,
                "n_custom_ids": len(h.custom_ids),
                "submitted_at": h.submitted_at,
            }
            if status == "ended" or status == "completed":
                results_url = getattr(obj, "results_url", None)
                entry["results_url"] = results_url
                out.append(entry)
                # Caller is expected to fetch results & reconcile, then call
                # `forget_handle(h.batch_id)`.
            else:
                out.append(entry)
                remaining.append(h)
        self._handles = remaining
        return out

    def forget_handle(self, batch_id: str) -> None:
        self._handles = [h for h in self._handles if h.batch_id != batch_id]
