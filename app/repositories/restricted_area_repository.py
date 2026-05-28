from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.restricted_areas import RestrictedArea


class RestrictedAreaRepository:
    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        filters = []
        if camera_id:
            filters.append(RestrictedArea.camera_id == camera_id)

        total = await db.scalar(
            select(func.count()).select_from(RestrictedArea).where(*filters)
        )
        result = await db.execute(
            select(RestrictedArea)
            .where(*filters)
            .order_by(RestrictedArea.timestamp.desc(), RestrictedArea.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
        return result.scalars().all(), int(total or 0)
