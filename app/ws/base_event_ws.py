"""Shared fan-out broadcaster for AI event WebSockets, with optional
per-camera filtering.

All four event streams (face / plate / restricted-area / mask) had a
byte-identical broadcaster; they now share this one class. A subscriber may
listen to EVERY camera (``camera_id=None`` — what the Live View wall does, it
shows all cameras) or to a SINGLE camera (``camera_id="cam-x"`` — what the
Recordings review page does, it reviews one camera at a time). A single-camera
client only receives frames whose ``payload["camera_id"]`` matches, so the
server does the filtering instead of shipping every camera's events to the
client to drop on the floor.

Threading: the AI persist path runs in the process_ai_service thread (its own
asyncio loop) while WebSocket clients live on FastAPI's main loop. ``publish``
is therefore thread-safe — it captures FastAPI's loop on first connect and hops
onto it via ``run_coroutine_threadsafe`` so the actual ``send_text`` runs where
the WebSocket object expects to be driven from.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Dict, Optional

from fastapi import WebSocket


class CameraFilteredBroadcaster:
    def __init__(self, name: str) -> None:
        # `name` only labels the shutdown log line so a skipped publish is
        # attributable to the right stream.
        self._name = name
        # ws -> camera filter. None means "all cameras"; a string means only
        # frames for that camera_id go to this client.
        self._clients: Dict[WebSocket, Optional[str]] = {}
        self._lock = asyncio.Lock()
        # FastAPI loop, bound on first connect. None means nobody is
        # listening yet, so `publish` is a cheap no-op.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self, ws: WebSocket, camera_id: Optional[str] = None) -> None:
        await ws.accept()
        self._loop = asyncio.get_running_loop()
        async with self._lock:
            # Empty string from a `?camera_id=` with no value means "all".
            self._clients[ws] = camera_id or None

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(ws, None)

    def has_clients(self, camera_id: Optional[str] = None) -> bool:
        """Có ai đang nghe camera này không — để chỗ gọi BỎ QUA việc dựng
        payload khi không ai xem. Đọc không khoá là cố ý: đây là đường nóng
        (mỗi khung hình), và đọc trượt một nhịp chỉ khiến gói thừa/thiếu đúng
        một khung."""
        if not self._clients:
            return False
        if camera_id is None:
            return True
        cam = str(camera_id)
        return any(want is None or want == cam for want in list(self._clients.values()))

    def publish(self, payload: dict) -> None:
        """Fire-and-forget from any thread. Safe to call before any client
        has connected (drops the message) and from inside the FastAPI loop
        (hops back onto itself, which is harmless)."""
        loop = self._loop
        if loop is None or not self._clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)
        except RuntimeError as exc:  # loop closed during shutdown
            print(f"{self._name} ws publish skipped: {exc}", file=sys.stderr)

    async def _broadcast(self, payload: dict) -> None:
        if not self._clients:
            return
        msg = json.dumps(payload, default=_default_json)
        cam = payload.get("camera_id")
        cam = str(cam) if cam is not None else None
        async with self._lock:
            stale: list = []
            for ws, want in self._clients.items():
                # want=None -> subscribed to all cameras; otherwise the frame's
                # camera must match this client's filter.
                if want is not None and want != cam:
                    continue
                try:
                    await ws.send_text(msg)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._clients.pop(ws, None)


def _default_json(value: Any):
    # SQLAlchemy may hand us datetimes or other non-JSON-native types; fall
    # back to str() so the broadcast never dies on a serialisation edge.
    return str(value)
