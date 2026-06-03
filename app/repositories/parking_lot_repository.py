from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking_lot import ParkingLot


class ParkingLotRepository:
    @staticmethod
    async def create(
        db: AsyncSession,
        face_camera_id: str,
        plate_camera_id: str,
        name: Optional[str] = None,
    ) -> ParkingLot:
        entry = ParkingLot(
            face_camera_id=face_camera_id,
            plate_camera_id=plate_camera_id,
            name=name,
        )
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def update(
        db: AsyncSession,
        entry: ParkingLot,
        face_camera_id: str,
        plate_camera_id: str,
        name: Optional[str] = None,
    ) -> ParkingLot:
        entry.face_camera_id = face_camera_id
        entry.plate_camera_id = plate_camera_id
        entry.name = name
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def get(db: AsyncSession, entry_id: int) -> Optional[ParkingLot]:
        result = await db.execute(
            select(ParkingLot).where(ParkingLot.id == entry_id)
        )
        return result.scalars().first()

    @staticmethod
    async def get_by_camera(
        db: AsyncSession, camera_id: str,
    ) -> Optional[ParkingLot]:
        # A camera id may appear in either slot; used to enforce that a camera
        # belongs to at most one parking lot.
        result = await db.execute(
            select(ParkingLot).where(
                or_(
                    ParkingLot.face_camera_id == camera_id,
                    ParkingLot.plate_camera_id == camera_id,
                )
            )
        )
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
            filters.append(ParkingLot.name.ilike(f"%{name}%"))

        total = await db.scalar(
            select(func.count()).select_from(ParkingLot).where(*filters)
        )
        result = await db.execute(
            select(ParkingLot)
            .where(*filters)
            .order_by(ParkingLot.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        return result.scalars().all(), int(total or 0)

    @staticmethod
    async def delete(db: AsyncSession, entry: ParkingLot) -> None:
        await db.delete(entry)
        await db.commit()

    @staticmethod
    async def list_all(db: AsyncSession):
        """Used at startup to prime the in-memory cache. Parking lots are
        few (one per gate), so an unbounded select is fine."""
        result = await db.execute(select(ParkingLot))
        return result.scalars().all()
