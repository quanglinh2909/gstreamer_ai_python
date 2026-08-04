from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event_face import EventFace
from app.models.identity import Identity
from app.repositories.event_repository_base import EventRepositoryBase


class EventFaceRepository(EventRepositoryBase):
    model = EventFace

    @classmethod
    async def list_paginated(
        cls,
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        """Như bản dùng chung, thêm TÊN người được khớp.

        Ghi đè vì phải outer-join sang bảng identity: sự kiện chỉ giữ
        identity_id mà bảng sự kiện thì hiện tên. Lọc/đếm/sắp vẫn lấy nguyên
        của lớp cơ sở.
        """
        filters = cls._filters(camera_id)
        total = await cls._count(db, filters)

        result = await db.execute(
            cls._ordered(filters, page, size)
            .add_columns(Identity.name)
            .outerjoin(Identity, EventFace.identity_id == Identity.id)
        )

        events = []
        for event, name in result.all():
            event.name = name
            events.append(event)
        return events, total
