"""Fan-out plate events to any number of WebSocket subscribers.

Mirrors face_event_ws — separate instance so a UI can subscribe to only
the events it cares about. The plate persist path runs in the
process_ai_service thread; WebSocket clients live on FastAPI's main loop.
`publish` is therefore thread-safe: it captures FastAPI's loop on first
connect and hops onto it via `run_coroutine_threadsafe` so the actual
`send_text` runs where the WebSocket object expects to be driven from."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Optional, Set

from fastapi import WebSocket


class PlateEventBroadcaster:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        # FastAPI loop, bound on first connect. None means nobody is
        # listening yet, so `publish` is a cheap no-op.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._loop = asyncio.get_running_loop()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    def publish(self, payload: dict) -> None:
        """Fire-and-forget from any thread."""
        loop = self._loop
        if loop is None or not self._clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
        except RuntimeError as exc:  # loop closed during shutdown
            print(f"plate ws publish skipped: {exc}", file=sys.stderr)

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload, default=_default_json)
        async with self._lock:
            stale: list = []
            for ws in self._clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._clients.discard(ws)


def _default_json(value: Any):
    return str(value)


plate_event_broadcaster = PlateEventBroadcaster()
