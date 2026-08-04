# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.event_dto import EventMaskResponse
from app.dto.face_mask_dto import FaceMaskDTO
from app.dto.pagination_dto import PageResponse
from app.services.face_mask_service import face_mask_service

router = APIRouter()
prefix = "/face-mask"
tags = ["Face Mask"]


@router.post("")
async def face_mask(
    req: FaceMaskDTO, db: AsyncSession = Depends(get_db),
):
    return await face_mask_service.face_mask(db, req)


@router.get("/events", response_model=PageResponse[EventMaskResponse])
async def list_face_mask_events(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    camera_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Lịch sử sự kiện khẩu trang — cùng dạng phân trang với ba loại AI kia."""
    items, total = await face_mask_service.list_events(db, page, size, camera_id)
    return PageResponse[EventMaskResponse].build(
        items=[EventMaskResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )

