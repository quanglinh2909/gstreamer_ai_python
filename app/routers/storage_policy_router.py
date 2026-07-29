# -*- coding: utf-8 -*-
"""API cấu hình tự dọn dung lượng + xem trạng thái đĩa/kích thước từng loại."""

import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.storage_cleanup_service import storage_cleanup_service

router = APIRouter()
prefix = "/storage-policy"
tags = ["Storage Policy"]


class StoragePolicyIn(BaseModel):
    enabled: Optional[bool] = None
    min_free_gb: Optional[float] = Field(None, gt=0)
    target_free_gb: Optional[float] = Field(None, gt=0)
    w_record: Optional[float] = Field(None, ge=0)
    w_event_face: Optional[float] = Field(None, ge=0)
    w_event_plate: Optional[float] = Field(None, ge=0)
    w_parking_lot_event: Optional[float] = Field(None, ge=0)
    w_restricted_area: Optional[float] = Field(None, ge=0)


_FIELDS = [
    "enabled", "min_free_gb", "target_free_gb", "w_record", "w_event_face",
    "w_event_plate", "w_parking_lot_event", "w_restricted_area",
]


async def _get_policy(db: AsyncSession) -> dict:
    row = (await db.execute(text(
        "SELECT id, enabled, min_free_gb, target_free_gb, w_record, w_event_face, "
        "w_event_plate, w_parking_lot_event, w_restricted_area, updated_at "
        "FROM storage_policy WHERE id = 1"
    ))).mappings().first()
    return dict(row) if row else None


@router.get("")
async def get_policy(db: AsyncSession = Depends(get_db)):
    """Cấu hình hiện tại."""
    return await _get_policy(db)


@router.put("")
async def update_policy(body: StoragePolicyIn, db: AsyncSession = Depends(get_db)):
    """Cập nhật cấu hình. Chỉ đổi các trường được gửi (partial update)."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if updates:
        set_sql = ", ".join(f"{k} = :{k}" for k in updates if k in _FIELDS)
        updates["ts"] = int(time.time())
        await db.execute(
            text(f"UPDATE storage_policy SET {set_sql}, updated_at = :ts WHERE id = 1"),
            updates,
        )
        await db.commit()
    return await _get_policy(db)


@router.get("/status")
async def get_status(db: AsyncSession = Depends(get_db)):
    """Đĩa (tổng/đã dùng/trống/%) + kích thước từng loại + chính sách — cho UI."""
    return await storage_cleanup_service.status(db)
