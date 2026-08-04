# -*- coding: utf-8 -*-
"""API CỤM CỔNG: nhiều camera cùng điều khiển một barrier.

Xem plate_gate_group.py để biết vì sao cụm là bảng riêng chứ không phải một
nhãn gắn lên camera.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.repositories.plate_gate_group_repository import PlateGateGroupRepository
from app.services.plate_white_list_settings_service import (
    plate_white_list_settings_service,
)

router = APIRouter()
prefix = "/plate-gate-groups"
tags = ["Plate Gate Groups"]


class PlateGateGroupIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    pre_time: int = Field(
        0, ge=0, le=3600,
        description="Giây chờ giữa 2 lần mở cho cùng một biển, tính CHUNG cả "
                    "cụm. Thay thế 'Chờ giữa 2 lần mở' của từng camera trong "
                    "cụm. 0 = mỗi biển chỉ mở được đúng một lần.",
    )
    camera_ids: Optional[List[str]] = Field(
        None,
        description="Danh sách ĐẦY ĐỦ camera thuộc cụm. Bỏ trống (null) = "
                    "không đụng tới thành viên hiện tại. Chỉ nhận camera đã "
                    "bật whitelist — cụm không tự bật barrier cho camera nào.",
    )


class PlateGateGroupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    pre_time: int
    camera_ids: List[str] = []


def _to_response(group) -> PlateGateGroupResponse:
    return PlateGateGroupResponse(
        id=group.id,
        name=group.name,
        pre_time=int(group.pre_time),
        # Đọc từ cache đã giải sẵn thay vì query lại: cache là thứ đường nóng
        # thực sự dùng, nên hiện đúng nó mới phát hiện được lệch DB/cache.
        camera_ids=sorted(
            plate_white_list_settings_service.cameras_in_group(group.id)
        ),
    )


async def _assert_name_free(db: AsyncSession, name: str, exclude_id=None) -> str:
    name = " ".join(name.split())
    if not name:
        raise HTTPException(status_code=400, detail="Ten cum khong duoc de trong")
    existing = await PlateGateGroupRepository.get_by_name(db, name)
    if existing is not None and existing.id != exclude_id:
        raise HTTPException(
            status_code=409, detail=f"Da co cum ten '{existing.name}'",
        )
    return name


@router.get("", response_model=List[PlateGateGroupResponse])
async def list_groups(db: AsyncSession = Depends(get_db)):
    groups = await PlateGateGroupRepository.list_all(db)
    return [_to_response(g) for g in groups]


@router.post("", response_model=PlateGateGroupResponse, status_code=201)
async def create_group(payload: PlateGateGroupIn, db: AsyncSession = Depends(get_db)):
    name = await _assert_name_free(db, payload.name)
    group = await PlateGateGroupRepository.create(
        db, {"name": name, "pre_time": payload.pre_time},
    )
    if payload.camera_ids is not None:
        await PlateGateGroupRepository.set_members(db, group.id, payload.camera_ids)
    # Nạp lại cache NGAY: cụm vừa tạo phải có hiệu lực từ khung hình kế tiếp,
    # không đợi restart.
    await plate_white_list_settings_service.load_all(db)
    return _to_response(group)


@router.put("/{group_id}", response_model=PlateGateGroupResponse)
async def update_group(
    group_id: int, payload: PlateGateGroupIn, db: AsyncSession = Depends(get_db),
):
    group = await PlateGateGroupRepository.get_by_id(db, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Cum khong ton tai")
    name = await _assert_name_free(db, payload.name, exclude_id=group_id)
    group = await PlateGateGroupRepository.update(
        db, group, {"name": name, "pre_time": payload.pre_time},
    )
    if payload.camera_ids is not None:
        await PlateGateGroupRepository.set_members(db, group_id, payload.camera_ids)
    await plate_white_list_settings_service.load_all(db)
    return _to_response(group)


@router.delete("/{group_id}", status_code=204)
async def delete_group(group_id: int, db: AsyncSession = Depends(get_db)):
    """Xoá cụm. Các camera trong cụm KHÔNG bị tắt barrier — chúng quay về dùng
    'Chờ giữa 2 lần mở' của riêng mình."""
    group = await PlateGateGroupRepository.get_by_id(db, group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Cum khong ton tai")
    await PlateGateGroupRepository.delete(db, group)
    await plate_white_list_settings_service.load_all(db)
