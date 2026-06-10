# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.identity_dto import IdentityResponse, IdentityWithFaceResponse
from app.dto.pagination_dto import PageResponse
from app.services.identity_service import identity_service

router = APIRouter()
prefix = "/identities"
tags = ["Identity"]


@router.post("", response_model=IdentityWithFaceResponse)
async def create_identity(
    name: str = Form(...),
    mac_bluetooth: Optional[str] = Form(None),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    return await identity_service.create_with_face(
        db=db,
        name=name,
        mac_bluetooth=mac_bluetooth,
        image=(image.filename, await image.read(), image.content_type),
    )


@router.put("/{identity_id}", response_model=IdentityWithFaceResponse)
async def update_identity(
    identity_id: int,
    name: Optional[str] = Form(None),
    mac_bluetooth: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    image_tuple = None
    if image is not None:
        image_tuple = (image.filename, await image.read(), image.content_type)
    return await identity_service.update_with_face(
        db=db,
        identity_id=identity_id,
        name=name,
        mac_bluetooth=mac_bluetooth,
        image=image_tuple,
    )


@router.get("", response_model=PageResponse[IdentityResponse])
async def list_identities(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    name: Optional[str] = Query(None, description="Substring match on name (case-insensitive)"),
    db: AsyncSession = Depends(get_db),
):
    items, total = await identity_service.list_paginated(db, page, size, name)
    return PageResponse[IdentityResponse].build(
        items=[IdentityResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        size=size,
    )


@router.get("/{identity_id}", response_model=IdentityResponse)
async def get_identity(identity_id: int, db: AsyncSession = Depends(get_db)):
    identity = await identity_service.get(db, identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    return IdentityResponse.model_validate(identity)


@router.delete("/{identity_id}", status_code=204)
async def delete_identity(identity_id: int, db: AsyncSession = Depends(get_db)):
    await identity_service.delete(db, identity_id)
    return Response(status_code=204)
