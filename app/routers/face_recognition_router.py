# -*- coding: utf-8 -*-

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.face_recognition_dto import FaceRecognitionDTO
from app.services.face_recognition_service import face_recognition_service

router = APIRouter()
prefix = "/face-recognition"
tags = ["Face Recognition"]


@router.post("")
async def face_recognition(req: FaceRecognitionDTO,db: AsyncSession = Depends(get_db),):
    return await face_recognition_service.face_recognition(db,req)
