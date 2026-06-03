from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity_plate import IdentityPlate


class IdentityPlateRepository:
    @staticmethod
    async def create(
        db: AsyncSession,
        identity_id: int,
        plate_number: str,
    ) -> IdentityPlate:
        entry = IdentityPlate(identity_id=identity_id, plate_number=plate_number)
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def update(
        db: AsyncSession,
        entry: IdentityPlate,
        plate_number: str,
    ) -> IdentityPlate:
        entry.plate_number = plate_number
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def get(db: AsyncSession, entry_id: int) -> Optional[IdentityPlate]:
        result = await db.execute(
            select(IdentityPlate).where(IdentityPlate.id == entry_id)
        )
        return result.scalars().first()

    @staticmethod
    async def get_by_identity_and_plate(
        db: AsyncSession, identity_id: int, plate_number: str,
    ) -> Optional[IdentityPlate]:
        # Case-insensitive lookup so the same plate can't be added twice to one
        # identity just because of capitalisation differences.
        result = await db.execute(
            select(IdentityPlate).where(
                IdentityPlate.identity_id == identity_id,
                func.upper(IdentityPlate.plate_number) == plate_number.upper(),
            )
        )
        return result.scalars().first()

    @staticmethod
    async def list_by_identity(
        db: AsyncSession, identity_id: int,
    ) -> List[IdentityPlate]:
        result = await db.execute(
            select(IdentityPlate)
            .where(IdentityPlate.identity_id == identity_id)
            .order_by(IdentityPlate.id.desc())
        )
        return result.scalars().all()

    @staticmethod
    async def delete(db: AsyncSession, entry: IdentityPlate) -> None:
        await db.delete(entry)
        await db.commit()

    @staticmethod
    async def list_all(db: AsyncSession) -> List[IdentityPlate]:
        """Used at startup to prime the in-memory cache so the AI pipeline can
        map a detected plate to its identity without hitting the DB."""
        result = await db.execute(select(IdentityPlate))
        return result.scalars().all()
