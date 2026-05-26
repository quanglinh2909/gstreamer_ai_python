# -*- coding: utf-8 -*-

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.services.plate_recognition_service import plate_recognition_service

router = APIRouter()
prefix = "/plate-recognition"
tags = ["Plate Recognition"]


@router.post("")
async def vehicle_ai(req: PlateRecognitionDTO, db: AsyncSession = Depends(get_db), ):
    return await plate_recognition_service.plate_recognition(db, req)
