from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event_plate import EventPlate


class EventPlateRepository:
    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        filters = []
        if camera_id:
            filters.append(EventPlate.camera_id == camera_id)

        total = await db.scalar(
            select(func.count()).select_from(EventPlate).where(*filters)
        )

        result = await db.execute(
            select(EventPlate)
            .where(*filters)
            .order_by(EventPlate.timestamp.desc(), EventPlate.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        return result.scalars().all(), int(total or 0)
