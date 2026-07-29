# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.ws.live_detection_ws import live_detection_broadcaster

router = APIRouter()
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/live-detections")
async def live_detections_ws(ws: WebSocket, camera_id: Optional[str] = Query(None)):
    """Một JSON mỗi khung hình AI xử lý, để vẽ khung đè lên video trực tiếp:

    {"camera_id": str, "job_id": str, "ai_type": str, "seq": int,
     "ts": float, "width": int, "height": int,
     "boxes": [{"x1": 0.12, "y1": 0.30, "x2": 0.25, "y2": 0.61,
                "score": 0.87, "class_id": 0, "label": "person", "tid": 12}]}

    Toạ độ CHUẨN HOÁ [0,1] theo khung hình AI nhận được, nên client vẽ được ở
    mọi kích thước hiển thị mà không cần biết độ phân giải camera. `boxes` rỗng
    nghĩa là khung này không phát hiện gì — client phải XOÁ khung đang vẽ.

    Nên mở kèm `?camera_id=` (chỉ nhận đúng camera đang xem): đây là luồng liên
    tục, không lọc thì nhận khung phát hiện của mọi camera. Một camera có thể
    chạy nhiều job AI cùng lúc, mỗi job là một dòng riêng — client gom theo
    `job_id` rồi vẽ hợp của các job.

    Server không đợi client gửi gì; chỉ đọc để phát hiện ngắt kết nối."""
    await live_detection_broadcaster.connect(ws, camera_id)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await live_detection_broadcaster.disconnect(ws)
