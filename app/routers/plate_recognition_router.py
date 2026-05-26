# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.event_dto import EventPlateResponse
from app.dto.pagination_dto import PageResponse
from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.services.plate_recognition_service import plate_recognition_service

router = APIRouter()
prefix = "/plate-recognition"
tags = ["Plate Recognition"]


@router.post("")
async def vehicle_ai(req: PlateRecognitionDTO, db: AsyncSession = Depends(get_db)):
    return await plate_recognition_service.plate_recognition(db, req)


@router.post("/test")
async def plate_recognition_test(
    image: UploadFile = File(...),
    primary_conf: float = Form(0.3),
    secondary_conf: float = Form(0.3),
):
    return await plate_recognition_service.test_inference(
        image=(image.filename, await image.read(), image.content_type),
        primary_conf=primary_conf,
        secondary_conf=secondary_conf,
    )


@router.get("/events", response_model=PageResponse[EventPlateResponse])
async def list_plate_events(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    camera_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    items, total = await plate_recognition_service.list_events(db, page, size, camera_id)
    return PageResponse[EventPlateResponse].build(
        items=[EventPlateResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )
