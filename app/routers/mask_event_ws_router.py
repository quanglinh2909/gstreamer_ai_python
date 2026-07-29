# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.ws.mask_event_ws import mask_event_broadcaster

router = APIRouter()
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/mask-events")
async def mask_events_ws(ws: WebSocket, camera_id: Optional[str] = Query(None)):
    """Streams one JSON message per face-mask (access-control) event:

    {"id": str|int, "camera_id": str, "confidence": float, "timestamp": int,
     "mask_status": "wearing_mask" | "not_wearing_mask" | "unknown",
     "image_full": "data:image/jpeg;base64,...",
     "image_crop": "data:image/jpeg;base64,..."}

    Field names match the recognition-event shape (image_full/image_crop) so
    the UI renders it through the same feed component; the image is inline
    base64 (mask events are not persisted to /uploads). The server drains
    incoming frames only so a client disconnect propagates as
    WebSocketDisconnect."""
    await mask_event_broadcaster.connect(ws, camera_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await mask_event_broadcaster.disconnect(ws)
