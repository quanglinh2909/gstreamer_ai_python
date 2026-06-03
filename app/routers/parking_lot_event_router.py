# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.pagination_dto import PageResponse
from app.dto.parking_lot_event_dto import ParkingLotEventResponse
from app.services.parking_lot_event_service import parking_lot_event_service

router = APIRouter()
prefix = "/parking-lot-events"
tags = ["Parking Lot Event"]


@router.get("", response_model=PageResponse[ParkingLotEventResponse])
async def list_parking_lot_events(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    name: Optional[str] = Query(
        None, description="Substring match on identity name (case-insensitive)",
    ),
    identity_id: Optional[int] = Query(None, description="Exact identity id"),
    plate_number: Optional[str] = Query(
        None, description="Substring match on plate number (case-insensitive)",
    ),
    db: AsyncSession = Depends(get_db),
):
    items, total = await parking_lot_event_service.list_paginated(
        db, page, size, name, identity_id, plate_number,
    )
    return PageResponse[ParkingLotEventResponse].build(
        items=[ParkingLotEventResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )
