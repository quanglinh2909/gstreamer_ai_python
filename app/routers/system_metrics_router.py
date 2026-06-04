# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.system_metrics_dto import (
    SystemMetricsCurrent,
    SystemMetricsHistory,
    SystemMetricsResponse,
)
from app.services.system_metrics_service import system_metrics_service

router = APIRouter()
prefix = "/system-metrics"
tags = ["System Metrics"]


@router.get("", response_model=SystemMetricsResponse)
async def get_system_metrics(
    limit: int = Query(
        500, ge=1, le=300000,
        description="Số bản ghi tối đa mỗi loại chỉ số (mới nhất trước).",
    ),
    from_ts: Optional[int] = Query(
        None, description="Epoch giây — chỉ lấy bản ghi có ts >= from_ts.",
    ),
    to_ts: Optional[int] = Query(
        None, description="Epoch giây — chỉ lấy bản ghi có ts <= to_ts.",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Trả về tất cả 6 chỉ số (cpu usage, nhiệt độ cpu, memory, load avg, npu,
    rga) trong một response duy nhất: `current` là giá trị hiện tại của từng
    chỉ số, `history` là lịch sử (giữ trong 1 tháng gần nhất)."""
    data = await system_metrics_service.fetch_all(db, limit, from_ts, to_ts)
    return SystemMetricsResponse(
        current=SystemMetricsCurrent.model_validate(data["current"]),
        history=SystemMetricsHistory.model_validate(data["history"]),
    )
