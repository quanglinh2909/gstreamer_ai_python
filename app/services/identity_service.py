import asyncio
import os
import shutil
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.identity_dto import IdentityWithFaceResponse
from app.repositories.face_vector_repository import FaceVectorRepository
from app.repositories.identity_repository import IdentityRepository
from app.services.face_recognition_service import UPLOADS_ROOT, face_recognition_service


class IdentityService:
    async def create_with_face(
        self,
        db: AsyncSession,
        name: str,
        image: tuple,
        mac_bluetooth: Optional[str] = None,
    ) -> IdentityWithFaceResponse:
        identity = await IdentityRepository.create(db, name, mac_bluetooth)
        try:
            face = await face_recognition_service.register_face(identity.id, image)
        except Exception:
            await db.delete(identity)
            await db.commit()
            raise
        identity.image_full = face.image_full
        identity.image_crop = face.image_crop
        await db.commit()
        await db.refresh(identity)
        return IdentityWithFaceResponse(
            id=identity.id,
            name=identity.name,
            mac_bluetooth=identity.mac_bluetooth,
            image_full=identity.image_full,
            image_crop=identity.image_crop,
            face=face,
        )

    async def update_with_face(
        self,
        db: AsyncSession,
        identity_id: int,
        name: Optional[str],
        image: Optional[tuple],
        mac_bluetooth: Optional[str] = None,
    ) -> IdentityWithFaceResponse:
        identity = await IdentityRepository.get(db, identity_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found")

        if name is not None or mac_bluetooth is not None:
            identity = await IdentityRepository.update(
                db, identity, name, mac_bluetooth
            )

        face = None
        if image is not None:
            FaceVectorRepository.delete_by_identity(identity_id)
            face = await face_recognition_service.register_face(identity_id, image)
            identity.image_full = face.image_full
            identity.image_crop = face.image_crop
            await db.commit()
            await db.refresh(identity)

        return IdentityWithFaceResponse(
            id=identity.id,
            name=identity.name,
            mac_bluetooth=identity.mac_bluetooth,
            image_full=identity.image_full,
            image_crop=identity.image_crop,
            face=face,
        )

    async def get(self, db: AsyncSession, identity_id: int):
        return await IdentityRepository.get(db, identity_id)

    async def list_paginated(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        name: Optional[str] = None,
    ):
        return await IdentityRepository.list_paginated(db, page, size, name)

    async def delete(self, db: AsyncSession, identity_id: int) -> None:
        identity = await IdentityRepository.get(db, identity_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        # Milvus vectors first — if it fails we keep the SQL row so a retry
        # can finish the cleanup. event_face rows are SET NULL via FK cascade.
        await asyncio.to_thread(FaceVectorRepository.delete_by_identity, identity_id)
        await IdentityRepository.delete(db, identity)
        folder = os.path.join(UPLOADS_ROOT, "identities", str(identity_id))
        await asyncio.to_thread(shutil.rmtree, folder, True)


identity_service = IdentityService()
