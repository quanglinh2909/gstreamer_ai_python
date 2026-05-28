from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plate_white_list import PlateWhiteList


class PlateWhiteListRepository:
    @staticmethod
    async def create(
        db: AsyncSession, plate_number: str, name: Optional[str] = None,
    ) -> PlateWhiteList:
        entry = PlateWhiteList(plate_number=plate_number, name=name)
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def update(
        db: AsyncSession,
        entry: PlateWhiteList,
        plate_number: str,
        name: Optional[str] = None,
    ) -> PlateWhiteList:
        entry.plate_number = plate_number
        entry.name = name
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def get(db: AsyncSession, entry_id: int) -> Optional[PlateWhiteList]:
        result = await db.execute(
            select(PlateWhiteList).where(PlateWhiteList.id == entry_id)
        )
        return result.scalars().first()

    @staticmethod
    async def get_by_plate(
        db: AsyncSession, plate_number: str,
    ) -> Optional[PlateWhiteList]:
        # Case-insensitive lookup so callers can validate uniqueness without
        # caring how the value was capitalised on the way in.
        result = await db.execute(
            select(PlateWhiteList).where(
                func.upper(PlateWhiteList.plate_number) == plate_number.upper()
            )
        )
        return result.scalars().first()

    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        plate_number: Optional[str] = None,
    ):
        filters = []
        if plate_number:
            filters.append(PlateWhiteList.plate_number.ilike(f"%{plate_number}%"))

        total = await db.scalar(
            select(func.count()).select_from(PlateWhiteList).where(*filters)
        )
        result = await db.execute(
            select(PlateWhiteList)
            .where(*filters)
            .order_by(PlateWhiteList.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        return result.scalars().all(), int(total or 0)

    @staticmethod
    async def delete(db: AsyncSession, entry: PlateWhiteList) -> None:
        await db.delete(entry)
        await db.commit()

    @staticmethod
    async def list_all(db: AsyncSession):
        """Used at startup to prime the in-memory cache. Whitelists are
        typically small (10s–1000s of rows) so an unbounded select is
        fine; bump to streaming/yield_per if this ever grows past ~100k."""
        result = await db.execute(select(PlateWhiteList))
        return result.scalars().all()
