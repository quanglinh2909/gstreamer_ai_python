# -*- coding: utf-8 -*-
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.ws.system_metrics_ws import system_metrics_broadcaster

router = APIRouter()
prefix = "/ws"
tags = ["WebSocket"]


@router.websocket("/system-metrics")
async def system_metrics_ws(ws: WebSocket):
    """Đẩy giá trị hiện tại của 6 chỉ số mỗi 10 giây (cùng nhịp với lưu DB).

    Mỗi message là một JSON:

    {"type": "system_metrics", "ts": int,
     "cpu_usage": {"usage_percent": float, "per_core": [float, ...]},
     "cpu_temperature": {"soc_c": float, ..., "npu_c": float},
     "memory": {"total_bytes": int, "used_bytes": int,
                "available_bytes": int, "percent": float},
     "disk": {"total_bytes": int, "used_bytes": int,
              "free_bytes": int, "percent": float},
     "load_avg": {"load1": float, "load5": float, "load15": float,
                  "cpu_count": int},
     "npu": {"load_percent": float|null, "core0": ..., "core1": ..., "core2": ...},
     "rga": {"load_percent": float|null, "core0": ..., "core1": ..., "core2": ...}}

    Frame đầu tiên gửi ngay khi kết nối (mẫu gần nhất đang cache), sau đó cứ
    10s một frame. Server không cần input — đọc frame đến chỉ để phát hiện
    client ngắt kết nối."""
    await system_metrics_broadcaster.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await system_metrics_broadcaster.disconnect(ws)
