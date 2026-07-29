# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.ws.restricted_area_event_ws import restricted_area_event_broadcaster

router = APIRouter()
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/restricted-area-events")
async def restricted_area_events_ws(ws: WebSocket, camera_id: Optional[str] = Query(None)):
    """Streams one JSON message per persisted restricted-area event:

    {"id": int, "camera_id": str, "class_id": int|null,
     "confidence": float, "timestamp": int,
     "image_full": "/uploads/...", "image_crop": "/uploads/..."}

    Field shape matches RestrictedAreaResponse from the REST list
    endpoint plus a `class_id` hint (which YOLO class triggered — 0 for
    person, 1 for bicycle, etc.) so a UI can show "Người vào vùng cấm"
    vs "Xe vào vùng cấm" without an extra lookup. The server doesn't
    expect any input — incoming frames are drained only so a client
    disconnect propagates as WebSocketDisconnect."""
    await restricted_area_event_broadcaster.connect(ws, camera_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await restricted_area_event_broadcaster.disconnect(ws)
