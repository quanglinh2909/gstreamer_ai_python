from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event_face import EventFace
from app.models.identity import Identity


class EventFaceRepository:
    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        filters = []
        if camera_id:
            filters.append(EventFace.camera_id == camera_id)

        total = await db.scalar(
            select(func.count()).select_from(EventFace).where(*filters)
        )

        result = await db.execute(
            select(EventFace, Identity.name)
            .outerjoin(Identity, EventFace.identity_id == Identity.id)
            .where(*filters)
            .order_by(EventFace.timestamp.desc(), EventFace.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )

        events = []
        for event, name in result.all():
            event.name = name
            events.append(event)
        return events, int(total or 0)
