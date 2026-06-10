from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity import Identity


class IdentityRepository:
    @staticmethod
    async def create(
        db: AsyncSession, name: str, mac_bluetooth: Optional[str] = None
    ) -> Identity:
        identity = Identity(name=name, mac_bluetooth=mac_bluetooth)
        db.add(identity)
        await db.commit()
        await db.refresh(identity)
        return identity

    @staticmethod
    async def update(
        db: AsyncSession,
        identity: Identity,
        name: Optional[str] = None,
        mac_bluetooth: Optional[str] = None,
    ) -> Identity:
        if name is not None:
            identity.name = name
        if mac_bluetooth is not None:
            identity.mac_bluetooth = mac_bluetooth
        await db.commit()
        await db.refresh(identity)
        return identity

    @staticmethod
    async def get(db: AsyncSession, identity_id: int) -> Optional[Identity]:
        result = await db.execute(select(Identity).where(Identity.id == identity_id))
        return result.scalars().first()

    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        name: Optional[str] = None,
    ):
        filters = []
        if name:
            # Case-insensitive substring match; ilike is supported by Postgres
            # and SQLite (via case-insensitive LIKE).
            filters.append(Identity.name.ilike(f"%{name}%"))

        total = await db.scalar(
            select(func.count()).select_from(Identity).where(*filters)
        )
        result = await db.execute(
            select(Identity)
            .where(*filters)
            .order_by(Identity.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        return result.scalars().all(), int(total or 0)

    @staticmethod
    async def delete(db: AsyncSession, identity: Identity) -> None:
        await db.delete(identity)
        await db.commit()
