# -*- coding: utf-8 -*-

from fastapi import APIRouter, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.core.database import get_db
from app.dto.camera_dto import CameraCreateDTO, CameraUpdateDTO
from app.services.camera_service import camera_service

router = APIRouter()
prefix = "/cameras"
tags = ["Camera"]


@router.post("")
async def create(req: CameraCreateDTO):
    return await HTTPXClient.post("/cameras", json=req.model_dump(exclude_none=True))


@router.put("/{id}")
async def update(req: CameraUpdateDTO, id: str):
    return await HTTPXClient.put(f"/cameras/{id}", json=req.model_dump(exclude_none=True))


@router.get("")
async def get_cameras(limit: int = 10, offset: int = 0):
    return await HTTPXClient.get("/cameras", params={"limit": limit, "offset": offset})


@router.delete("/{id}")
async def delete(id: str, db: AsyncSession = Depends(get_db)):
    return await camera_service.delete_cammera(db, id)


@router.get("/{camera_id}/snapshot")
async def snapshot(camera_id: str):
    content, content_type = await HTTPXClient.get(
        f"/cameras/{camera_id}/snapshot", raw=True
    )
    return Response(content=content, media_type=content_type)

@router.get("/{camera_id}/config-ai")
async def get_by_camera_id(camera_id: str,db: AsyncSession = Depends(get_db)):
    return await camera_service.get_by_camera_id(db, camera_id)