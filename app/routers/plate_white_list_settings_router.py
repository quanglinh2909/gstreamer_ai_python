# -*- coding: utf-8 -*-
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.plate_white_list_settings_dto import (
    PlateWhiteListSettingsResponse,
    PlateWhiteListSettingsUpdate,
)
from app.services.plate_white_list_settings_service import (
    plate_white_list_settings_service,
)

router = APIRouter()
prefix = "/plate-white-list-settings"
tags = ["Plate White List Settings"]


@router.get("", response_model=List[PlateWhiteListSettingsResponse])
async def list_settings(db: AsyncSession = Depends(get_db)):
    """Danh sách camera ĐÃ bật whitelist/barrier. Camera không có trong danh
    sách này bị bỏ qua hoàn toàn ở nhánh whitelist."""
    rows = await plate_white_list_settings_service.list_all(db)
    return [PlateWhiteListSettingsResponse.model_validate(r) for r in rows]


@router.get("/{camera_id}", response_model=PlateWhiteListSettingsResponse)
async def get_settings(camera_id: str, db: AsyncSession = Depends(get_db)):
    """404 khi camera chưa cấu hình — không trả về giá trị mặc định giả, vì
    lúc đó nhánh whitelist đang TẮT chứ không phải đang chạy với giá trị nào
    đó. Form cấu hình bắt 404 rồi hiển thị trạng thái "chưa bật" kèm giá trị
    gợi ý mặc định của PlateWhiteListSettingsUpdate."""
    entry = await plate_white_list_settings_service.get(db, camera_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} chua bat whitelist/barrier",
        )
    return PlateWhiteListSettingsResponse.model_validate(entry)


@router.put("/{camera_id}", response_model=PlateWhiteListSettingsResponse)
async def upsert_settings(
    camera_id: str,
    payload: PlateWhiteListSettingsUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Tạo mới hoặc ghi đè cấu hình của camera. Cache được cập nhật ngay nên
    biển đọc ở frame kế tiếp đã dùng giá trị mới, không cần restart."""
    entry = await plate_white_list_settings_service.upsert(
        db, camera_id, payload.model_dump(),
    )
    return PlateWhiteListSettingsResponse.model_validate(entry)


@router.delete("/{camera_id}", status_code=204)
async def delete_settings(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Xoá cấu hình riêng → camera quay về giá trị mặc định."""
    await plate_white_list_settings_service.delete(db, camera_id)
    return Response(status_code=204)
