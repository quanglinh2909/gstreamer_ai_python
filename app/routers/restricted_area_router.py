# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.event_dto import RestrictedAreaResponse
from app.dto.pagination_dto import PageResponse
from app.dto.restricted_area_dto import RestrictedAreaDTO
from app.services.restricted_area_service import restricted_area_service

router = APIRouter()
prefix = "/restricted-area"
tags = ["Restricted Area"]


@router.post("")
async def restricted_area(
    req: RestrictedAreaDTO, db: AsyncSession = Depends(get_db),
):
    return await restricted_area_service.restricted_area(db, req)


@router.get("/settings")
async def restricted_area_settings(
    cameraId: str = Query(..., description="Camera cần xem cấu hình model/lớp."),
    db: AsyncSession = Depends(get_db),
):
    """Model + class filter đang áp cho camera (kèm giá trị mặc định) — giao
    diện dùng để hiển thị đúng lựa chọn hiện tại."""
    return await restricted_area_service.get_settings(db, cameraId)


@router.post("/test")
async def restricted_area_test(
    image: UploadFile = File(...),
    primary_conf: float = Form(0.3),
    secondary_conf: float = Form(0.3),
):
    return await restricted_area_service.test_inference(
        image=(image.filename, await image.read(), image.content_type),
        primary_conf=primary_conf,
        secondary_conf=secondary_conf,
    )


@router.get("/events", response_model=PageResponse[RestrictedAreaResponse])
async def list_restricted_area_events(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    camera_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    items, total = await restricted_area_service.list_events(
        db, page, size, camera_id,
    )
    return PageResponse[RestrictedAreaResponse].build(
        items=[RestrictedAreaResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )
