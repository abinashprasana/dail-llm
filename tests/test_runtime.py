import asyncio

import pytest

from dail_llm.api.runtime import BusyError, InferenceGate, RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_returns_retry_after():
    limiter = RateLimiter(requests=2, window_seconds=30)
    assert await limiter.allow("client") == (True, 0)
    assert await limiter.allow("client") == (True, 0)
    allowed, retry_after = await limiter.allow("client")
    assert allowed is False
    assert retry_after > 0


@pytest.mark.asyncio
async def test_gate_rejects_when_active_and_queue_full():
    gate = InferenceGate(concurrency=1, max_queue=0)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def occupy():
        async with gate.slot():
            entered.set()
            await release.wait()

    task = asyncio.create_task(occupy())
    await entered.wait()
    with pytest.raises(BusyError):
        async with gate.slot():
            pass
    release.set()
    await task
