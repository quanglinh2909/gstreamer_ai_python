from typing import Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.identity_dto import IdentityWithFaceResponse
from app.repositories.face_vector_repository import FaceVectorRepository
from app.repositories.identity_repository import IdentityRepository
from app.services.face_recognition_service import face_recognition_service


class IdentityService:
    async def create_with_face(
        self, db: AsyncSession, name: str, image: tuple
    ) -> IdentityWithFaceResponse:
        identity = await IdentityRepository.create(db, name)
        try:
            face = await face_recognition_service.register_face(identity.id, image)
        except Exception:
            await db.delete(identity)
            await db.commit()
            raise
        return IdentityWithFaceResponse(id=identity.id, name=identity.name, face=face)

    async def update_with_face(
        self,
        db: AsyncSession,
        identity_id: int,
        name: Optional[str],
        image: Optional[tuple],
    ) -> IdentityWithFaceResponse:
        identity = await IdentityRepository.get(db, identity_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found")

        if name is not None:
            identity = await IdentityRepository.update(db, identity, name)

        face = None
        if image is not None:
            FaceVectorRepository.delete_by_identity(identity_id)
            face = await face_recognition_service.register_face(identity_id, image)

        return IdentityWithFaceResponse(id=identity.id, name=identity.name, face=face)

    async def get(self, db: AsyncSession, identity_id: int):
        return await IdentityRepository.get(db, identity_id)

    async def list_all(self, db: AsyncSession):
        return await IdentityRepository.list_all(db)


identity_service = IdentityService()
