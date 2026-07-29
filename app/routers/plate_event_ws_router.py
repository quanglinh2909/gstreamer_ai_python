# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.ws.plate_event_ws import plate_event_broadcaster

router = APIRouter()
# Explicit /ws prefix to match the face WS endpoint convention. Auto
# prefix would have been /plate-event-ws which reads worse.
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/plate-events")
async def plate_events_ws(ws: WebSocket, camera_id: Optional[str] = Query(None)):
    """Streams one JSON message per persisted plate event:

    {"id": int, "camera_id": str, "plate_number": str,
     "whitelisted": bool, "confidence": float, "timestamp": int,
     "image_full": "/uploads/...", "image_crop": "/uploads/..."}

    Field shape matches EventPlateResponse from the REST list endpoint
    (plus a `whitelisted` flag computed on the fly), so a UI can render
    WS pushes through the same component as paginated history. The
    server doesn't expect any input — incoming frames are drained only
    so a client disconnect propagates as WebSocketDisconnect."""
    await plate_event_broadcaster.connect(ws, camera_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await plate_event_broadcaster.disconnect(ws)
