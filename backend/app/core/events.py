"""A tiny in-process async pub/sub bus.

The worker, arbiter and backends *publish* events; the WebSocket endpoint
*subscribes* and forwards them to the browser. Decoupling these means no part of
the GPU pipeline knows or cares about HTTP.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import time
from typing import Any

from .enums import EventType


class Event(dict):
    """A plain dict with a stable shape: ``{type, ts, **data}``."""

    def __init__(self, type: EventType | str, **data: Any) -> None:
        super().__init__(
            type=type.value if isinstance(type, EventType) else type,
            ts=time.time(),
            **data,
        )


class EventBus:
    def __init__(self, *, max_queue: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._max_queue = max_queue

    async def publish(self, event: Event) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest so live progress stays current.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    def emit(self, type: EventType | str, **data: Any) -> None:
        """Fire-and-forget publish from sync or async contexts."""
        event = Event(type, **data)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(self.publish(event))
        else:
            # No loop (rare in our async-only app): publish synchronously best-effort.
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[Event]]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._max_queue)
        self._subscribers.add(q)
        try:
            yield q
        finally:
            self._subscribers.discard(q)
