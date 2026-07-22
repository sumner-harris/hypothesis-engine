# Modified from the original work.
"""Tests for the in-memory event bus / SSE fanout."""

from __future__ import annotations

import asyncio

import pytest

from hypothesis_engine.orchestrator.events import Event, EventBus


@pytest.mark.asyncio
async def test_subscribe_receives_publish() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def reader() -> None:
        async for ev in bus.subscribe("ses_a"):
            received.append(ev)
            if len(received) >= 2:
                break

    task = asyncio.create_task(reader())
    # Give the subscriber a moment to register
    await asyncio.sleep(0.05)
    await bus.publish("ses_a", "match_complete", {"x": 1})
    await bus.publish("ses_a", "match_complete", {"x": 2})
    await asyncio.wait_for(task, timeout=2.0)
    assert [ev.payload["x"] for ev in received] == [1, 2]


@pytest.mark.asyncio
async def test_publishes_isolated_by_session() -> None:
    bus = EventBus()
    got_a: list[Event] = []

    async def reader() -> None:
        async for ev in bus.subscribe("ses_a"):
            got_a.append(ev)
            if got_a:
                break

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)
    await bus.publish("ses_b", "match_complete", {"x": "other"})
    await bus.publish("ses_a", "match_complete", {"x": "mine"})
    await asyncio.wait_for(task, timeout=2.0)
    assert len(got_a) == 1
    assert got_a[0].payload["x"] == "mine"


@pytest.mark.asyncio
async def test_unsubscribe_via_aclosing() -> None:
    """Deterministic unsubscribe using contextlib.aclosing."""
    import contextlib

    bus = EventBus()

    async def reader() -> None:
        async with contextlib.aclosing(bus.subscribe("ses_x")) as gen:
            async for _ in gen:
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.05)
    assert bus.subscriber_count("ses_x") == 1
    await bus.publish("ses_x", "hello", {})
    await asyncio.wait_for(task, timeout=2.0)
    assert bus.subscriber_count("ses_x") == 0


@pytest.mark.asyncio
async def test_publish_never_blocks_on_a_stuck_subscriber() -> None:
    """A subscriber that never drains its queue must not block the bus.

    Regression test: a previous implementation used `await q.put(ev)` after a
    `qsize()` check, which could still block if the queue filled between the
    check and the put. The fix uses `put_nowait` with a drop-oldest policy.
    """
    import contextlib

    bus = EventBus(max_buffer=4)

    # `subscribe()` registers its queue inside the generator body. We need to
    # drive __anext__ once so the queue exists, then cancel that task so the
    # queue is never drained.
    sub_gen = bus.subscribe("ses_z")
    pull_task = asyncio.create_task(sub_gen.__anext__())
    # Yield so subscribe() can register
    for _ in range(10):
        if bus.subscriber_count("ses_z") >= 1:
            break
        await asyncio.sleep(0.01)
    assert bus.subscriber_count("ses_z") == 1

    # Drain the first event ourselves so pull_task returns; from now on the
    # queue accumulates with no reader.
    await bus.publish("ses_z", "warmup", {})
    await asyncio.wait_for(pull_task, timeout=0.5)

    # Now publish far more than max_buffer; each call must return promptly.
    for i in range(50):
        await asyncio.wait_for(
            bus.publish("ses_z", "noisy", {"i": i}), timeout=0.25
        )

    # Clean up by closing the generator.
    with contextlib.suppress(Exception):
        await sub_gen.aclose()
