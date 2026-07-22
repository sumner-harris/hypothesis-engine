# Modified from the original work.
"""Tournament match + Elo update models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

MatchMode = Literal["pairwise", "debate", "batch", "invalid"]
Winner = Literal["a", "b"]


class TournamentMatch(BaseModel):
    id: str
    session_id: str
    created_at: datetime
    hyp_a: str
    hyp_b: str
    mode: MatchMode
    winner: Winner | None
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float | None = None
    elo_b_after: float | None = None
    rationale: str | None = None
    transcript_id: str | None = None
    similarity: float | None = None
    prompt1_hyp_id: str | None = None
    prompt2_hyp_id: str | None = None
    prompt1_side: Literal["a", "b"] | None = None
    prompt2_side: Literal["a", "b"] | None = None
    winner_prompt_position: int | None = None
    prompt1_chars: int | None = None
    prompt2_chars: int | None = None
    prompt_order_key: str | None = None


class EloJournalEntry(BaseModel):
    """Append-only ledger entry. UNIQUE(match_id) makes Elo updates idempotent."""

    update_id: str
    match_id: str
    hyp_a: str
    hyp_b: str
    winner: Winner
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float
    applied_at: int                      # epoch ms
