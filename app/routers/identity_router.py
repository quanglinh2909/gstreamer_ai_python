# -*- coding: utf-8 -*-
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dto.identity_dto import IdentityResponse, IdentityWithFaceResponse
from app.services.identity_service import identity_service

router = APIRouter()
prefix = "/identities"
tags = ["Identity"]


@router.post("", response_model=IdentityWithFaceResponse)
async def create_identity(
    name: str = Form(...),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    return await identity_service.create_with_face(
        db=db,
        name=name,
        image=(image.filename, await image.read(), image.content_type),
    )


@router.put("/{identity_id}", response_model=IdentityWithFaceResponse)
async def update_identity(
    identity_id: int,
    name: Optional[str] = Form(None),
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
        image=image_tuple,
    )


@router.get("", response_model=List[IdentityResponse])
async def list_identities(db: AsyncSession = Depends(get_db)):
    items = await identity_service.list_all(db)
    return [IdentityResponse.model_validate(i) for i in items]


@router.get("/{identity_id}", response_model=IdentityResponse)
async def get_identity(identity_id: int, db: AsyncSession = Depends(get_db)):
    identity = await identity_service.get(db, identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    return IdentityResponse.model_validate(identity)
