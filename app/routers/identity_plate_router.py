# -*- coding: utf-8 -*-
from typing import List

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.identity_plate_dto import (
    IdentityPlateCreate,
    IdentityPlateResponse,
    IdentityPlateUpdate,
)
from app.services.identity_plate_service import identity_plate_service

router = APIRouter()
prefix = "/identities"
tags = ["Identity Plate"]


@router.get("/{identity_id}/plates", response_model=List[IdentityPlateResponse])
async def list_identity_plates(
    identity_id: int, db: AsyncSession = Depends(get_db),
):
    items = await identity_plate_service.list_by_identity(db, identity_id)
    return [IdentityPlateResponse.model_validate(i) for i in items]


@router.post(
    "/{identity_id}/plates", response_model=IdentityPlateResponse, status_code=201,
)
async def create_identity_plate(
    identity_id: int,
    payload: IdentityPlateCreate,
    db: AsyncSession = Depends(get_db),
):
    entry = await identity_plate_service.create(
        db, identity_id, payload.plate_number,
    )
    return IdentityPlateResponse.model_validate(entry)


@router.put("/plates/{plate_id}", response_model=IdentityPlateResponse)
async def update_identity_plate(
    plate_id: int,
    payload: IdentityPlateUpdate,
    db: AsyncSession = Depends(get_db),
):
    entry = await identity_plate_service.update(
        db, plate_id, payload.plate_number,
    )
    return IdentityPlateResponse.model_validate(entry)


@router.delete("/plates/{plate_id}", status_code=204)
async def delete_identity_plate(
    plate_id: int, db: AsyncSession = Depends(get_db),
):
    await identity_plate_service.delete(db, plate_id)
    return Response(status_code=204)
