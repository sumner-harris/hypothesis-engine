"""Transcript record — one row per LLM call."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Transcript(BaseModel):
    id: str
    session_id: str
    task_id: str | None
    agent: str
    action: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    started_at: datetime
    finished_at: datetime
    artifact_path: str       # relative under data_dir; full messages[] on disk
