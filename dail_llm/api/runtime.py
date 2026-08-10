"""Concurrency and rate controls for the public inference API."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager


class BusyError(RuntimeError):
    pass


class InferenceGate:
    def __init__(self, concurrency: int, max_queue: int):
        self._semaphore = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency
        self._max_queue = max_queue
        self._active = 0
        self._waiting = 0
        self._lock = asyncio.Lock()

    @property
    def snapshot(self) -> dict[str, int]:
        return {"active": self._active, "waiting": self._waiting}

    @asynccontextmanager
    async def slot(self):
        async with self._lock:
            if self._active >= self._concurrency and self._waiting >= self._max_queue:
                raise BusyError("The model is at capacity. Try again shortly.")
            self._waiting += 1

        acquired = False
        try:
            await self._semaphore.acquire()
            acquired = True
            async with self._lock:
                self._waiting -= 1
                self._active += 1
            yield
        finally:
            if acquired:
                async with self._lock:
                    self._active -= 1
                self._semaphore.release()
            else:
                async with self._lock:
                    self._waiting = max(0, self._waiting - 1)


class RateLimiter:
    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
            return True, 0
