# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.pagination_dto import PageResponse
from app.dto.parking_lot_dto import (
    ParkingLotCreate,
    ParkingLotResponse,
    ParkingLotUpdate,
)
from app.services.parking_lot_service import parking_lot_service

router = APIRouter()
prefix = "/parking-lots"
tags = ["Parking Lot"]


@router.post("", response_model=ParkingLotResponse, status_code=201)
async def create_parking_lot(
    payload: ParkingLotCreate,
    db: AsyncSession = Depends(get_db),
):
    entry = await parking_lot_service.create(
        db, payload.face_camera_id, payload.plate_camera_id, payload.name,
    )
    return ParkingLotResponse.model_validate(entry)


@router.put("/{entry_id}", response_model=ParkingLotResponse)
async def update_parking_lot(
    entry_id: int,
    payload: ParkingLotUpdate,
    db: AsyncSession = Depends(get_db),
):
    entry = await parking_lot_service.update(
        db, entry_id, payload.face_camera_id, payload.plate_camera_id, payload.name,
    )
    return ParkingLotResponse.model_validate(entry)


@router.get("", response_model=PageResponse[ParkingLotResponse])
async def list_parking_lots(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    name: Optional[str] = Query(
        None, description="Substring match on name (case-insensitive)",
    ),
    db: AsyncSession = Depends(get_db),
):
    items, total = await parking_lot_service.list_paginated(db, page, size, name)
    return PageResponse[ParkingLotResponse].build(
        items=[ParkingLotResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/{entry_id}", response_model=ParkingLotResponse)
async def get_parking_lot(entry_id: int, db: AsyncSession = Depends(get_db)):
    entry = await parking_lot_service.get(db, entry_id)
    return ParkingLotResponse.model_validate(entry)


@router.delete("/{entry_id}", status_code=204)
async def delete_parking_lot(entry_id: int, db: AsyncSession = Depends(get_db)):
    await parking_lot_service.delete(db, entry_id)
    return Response(status_code=204)
