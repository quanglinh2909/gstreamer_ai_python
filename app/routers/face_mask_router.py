# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.face_mask_dto import FaceMaskDTO
from app.services.face_mask_service import face_mask_service

router = APIRouter()
prefix = "/face-mask"
tags = ["Face Mask"]


@router.post("")
async def face_mask(
    req: FaceMaskDTO, db: AsyncSession = Depends(get_db),
):
    return await face_mask_service.face_mask(db, req)

