# -*- coding: utf-8 -*-
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.ai_dto import AIEnabledCountResponse
from app.services.ai_config_service import ai_config_service

router = APIRouter()
prefix = "/ai"
tags = ["AI"]


@router.get("/enabled-count", response_model=AIEnabledCountResponse)
async def enabled_count(db: AsyncSession = Depends(get_db)):
    """Trả về số lượng AI đang bật cho từng loại AI (plate_recognition,
    face_recognition, restricted_area) cùng tổng số. Mỗi AIConfig là một AI
    đang bật trên một camera."""
    return await ai_config_service.enabled_count(db)
