from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity import Identity


class IdentityRepository:
    @staticmethod
    async def create(db: AsyncSession, name: str) -> Identity:
        identity = Identity(name=name)
        db.add(identity)
        await db.commit()
        await db.refresh(identity)
        return identity

    @staticmethod
    async def update(db: AsyncSession, identity: Identity, name: str) -> Identity:
        identity.name = name
        await db.commit()
        await db.refresh(identity)
        return identity

    @staticmethod
    async def get(db: AsyncSession, identity_id: int) -> Optional[Identity]:
        result = await db.execute(select(Identity).where(Identity.id == identity_id))
        return result.scalars().first()

    @staticmethod
    async def list_all(db: AsyncSession):
        result = await db.execute(select(Identity).order_by(Identity.id.desc()))
        return result.scalars().all()
