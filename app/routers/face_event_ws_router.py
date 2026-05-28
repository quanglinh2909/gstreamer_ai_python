# -*- coding: utf-8 -*-
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.face_event_ws import face_event_broadcaster

router = APIRouter()
# Explicit prefix — the auto-prefix would have been /face-event-ws which
# reads worse than the conventional /ws root for sockets.
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/face-events")
async def face_events_ws(ws: WebSocket):
    """Streams one JSON message per persisted face event:

    {"id": int, "camera_id": str, "identity_id": int|null,
     "name": str|null, "confidence": float, "timestamp": int,
     "image_full": "/uploads/...", "image_crop": "/uploads/..."}

    Field shape matches EventFaceResponse from the REST list endpoint so
    a UI can render WS pushes through the same component as paginated
    history. The server doesn't expect any input — incoming frames are
    drained only so a client disconnect propagates as WebSocketDisconnect."""
    await face_event_broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await face_event_broadcaster.disconnect(ws)
