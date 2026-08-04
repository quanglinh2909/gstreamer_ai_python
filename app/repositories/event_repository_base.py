# -*- coding: utf-8 -*-
"""Truy vấn danh sách dùng chung cho mọi bảng sự kiện AI.

Bốn repository sự kiện chỉ khác nhau đúng ở CÁI BẢNG: cùng lọc theo camera,
cùng đếm tổng, cùng sắp mới-nhất-trước rồi phân trang. Đặt chung ở đây để thêm
một loại sự kiện chỉ còn là khai `model = ...`.

Lớp con nào cần hơn thế (khuôn mặt phải join sang bảng người để lấy tên) thì
ghi đè `list_paginated` — vẫn kế thừa được `_filters`/`_count` nên phần chung
không bị chép lại.
"""

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


class EventRepositoryBase:
    #: Lớp model kế thừa AiEventMixin.
    model = None

    @classmethod
    def _filters(cls, camera_id: Optional[str]):
        return [cls.model.camera_id == camera_id] if camera_id else []

    @classmethod
    async def _count(cls, db: AsyncSession, filters) -> int:
        total = await db.scalar(
            select(func.count()).select_from(cls.model).where(*filters)
        )
        return int(total or 0)

    @classmethod
    def _ordered(cls, filters, page: int, size: int):
        # timestamp DESC rồi id DESC: nhiều sự kiện rơi vào cùng một giây (một
        # người bước qua sinh vài lần báo), sắp theo mỗi timestamp thì thứ tự
        # giữa chúng do Postgres tuỳ ý — phân trang sẽ lặp/bỏ sót hàng.
        return (
            select(cls.model)
            .where(*filters)
            .order_by(cls.model.timestamp.desc(), cls.model.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )

    @classmethod
    async def list_paginated(
        cls,
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        filters = cls._filters(camera_id)
        total = await cls._count(db, filters)
        result = await db.execute(cls._ordered(filters, page, size))
        return result.scalars().all(), total
