"""Thong tin de trinh duyet mo duoc phien MoQ.

Trinh phat PHAI hoi endpoint nay truoc khi ket noi: chung chi WebTransport tu
xoay vong 13 ngay mot lan (xem app/moq/cert.py) nen ma bam khong the nhung
cung trong ma nguon frontend.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.moq.server import moq_server

router = APIRouter()
prefix = "/moq"
tags = ["MoQ"]


@router.get("/info")
def moq_info(request: Request):
    if not moq_server.info:
        return JSONResponse(
            {"enabled": False,
             "reason": "may chu MoQ chua san sang" if settings.MOQ_ENABLED
                       else "MoQ dang tat trong cau hinh"},
            status_code=503,
        )
    info = dict(moq_server.info)
    info["enabled"] = True
    # `host` CHI dat khi nguoi van hanh cau hinh tuong minh. KHONG duoc suy ra
    # tu Host cua yeu cau: trinh duyet goi /moq/info qua proxy API cua Next,
    # nen o day ta thay Host = "127.0.0.1:8010" — dia chi cua chinh may chu,
    # vo nghia voi nguoi xem. De trong thi trinh phat dung host cua trang dang
    # mo, dung cho ca LAN lan ten mien.
    #
    # Chi can dat MOQ_PUBLIC_HOST khi QUIC nam o dia chi KHAC voi web (vd web
    # qua nginx/proxy o mot may, con cong UDP forward toi mot dia chi khac).
    info["host"] = settings.MOQ_PUBLIC_HOST or ""
    info["port"] = settings.MOQ_PUBLIC_PORT or info["port"]
    # Chi de nguoi doc/gõ lệnh kiểm tra nhìn cho tiện, trinh phat khong dung.
    shown = info["host"] or (request.headers.get("host", "") or "").split(":")[0]
    info["url"] = f"https://{shown}:{info['port']}{info['path']}"
    return info
