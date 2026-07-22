# Modified from the original work.
"""Tests for the batch pool (no Anthropic calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hypothesis_engine.config import Config
from hypothesis_engine.llm.batch import BatchedMatch, BatchPool


def _match(custom_id: str, session_id: str = "ses_t") -> BatchedMatch:
    return BatchedMatch(
        custom_id=custom_id, session_id=session_id,
        hyp_a="hyp_a", hyp_b="hyp_b",
        prompt="...", model="claude-sonnet-4-6", system="rules",
    )


@pytest.mark.asyncio
async def test_no_submit_below_min_size(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    pool = BatchPool(Config())
    pool.enqueue(_match("m1"))
    pool.enqueue(_match("m2"))
    # min_size=4 default → no submit
    handle = await pool.submit_batch(min_size=4)
    assert handle is None
    assert pool.pending_count == 2


@pytest.mark.asyncio
async def test_submit_calls_sdk_when_min_met(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    pool = BatchPool(Config())
    # Mock the SDK surface: client.messages.batches.create
    fake_batch = MagicMock(id="bat_123")
    pool._client = MagicMock()
    pool._client.messages = MagicMock()
    pool._client.messages.batches = MagicMock(
        create=AsyncMock(return_value=fake_batch),
    )
    for i in range(5):
        pool.enqueue(_match(f"m{i}"))
    h = await pool.submit_batch(min_size=4)
    assert h is not None
    assert h.batch_id == "bat_123"
    assert len(h.custom_ids) == 5
    assert pool.pending_count == 0
    assert pool.inflight_count == 1


@pytest.mark.asyncio
async def test_submit_failure_requeues(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    pool = BatchPool(Config())
    pool._client = MagicMock()
    pool._client.messages = MagicMock()
    pool._client.messages.batches = MagicMock(
        create=AsyncMock(side_effect=RuntimeError("boom")),
    )
    for i in range(5):
        pool.enqueue(_match(f"m{i}"))
    h = await pool.submit_batch(min_size=4)
    assert h is None
    # Items must be requeued so we don't lose work.
    assert pool.pending_count == 5
