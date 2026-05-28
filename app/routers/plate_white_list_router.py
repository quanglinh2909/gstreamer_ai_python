# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.pagination_dto import PageResponse
from app.dto.plate_white_list_dto import (
    PlateWhiteListCreate,
    PlateWhiteListResponse,
    PlateWhiteListUpdate,
)
from app.services.plate_white_list_service import plate_white_list_service

router = APIRouter()
prefix = "/plate-white-list"
tags = ["Plate White List"]


@router.post("", response_model=PlateWhiteListResponse, status_code=201)
async def create_entry(
    payload: PlateWhiteListCreate,
    db: AsyncSession = Depends(get_db),
):
    entry = await plate_white_list_service.create(
        db, payload.plate_number, payload.name,
    )
    return PlateWhiteListResponse.model_validate(entry)


@router.put("/{entry_id}", response_model=PlateWhiteListResponse)
async def update_entry(
    entry_id: int,
    payload: PlateWhiteListUpdate,
    db: AsyncSession = Depends(get_db),
):
    entry = await plate_white_list_service.update(
        db, entry_id, payload.plate_number, payload.name,
    )
    return PlateWhiteListResponse.model_validate(entry)


@router.get("", response_model=PageResponse[PlateWhiteListResponse])
async def list_entries(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    plate_number: Optional[str] = Query(
        None, description="Substring match on plate number (case-insensitive)",
    ),
    db: AsyncSession = Depends(get_db),
):
    items, total = await plate_white_list_service.list_paginated(
        db, page, size, plate_number,
    )
    return PageResponse[PlateWhiteListResponse].build(
        items=[PlateWhiteListResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/{entry_id}", response_model=PlateWhiteListResponse)
async def get_entry(entry_id: int, db: AsyncSession = Depends(get_db)):
    entry = await plate_white_list_service.get(db, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Whitelist entry not found")
    return PlateWhiteListResponse.model_validate(entry)


@router.delete("/{entry_id}", status_code=204)
async def delete_entry(entry_id: int, db: AsyncSession = Depends(get_db)):
    await plate_white_list_service.delete(db, entry_id)
    return Response(status_code=204)
