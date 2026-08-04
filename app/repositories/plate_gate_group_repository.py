# -*- coding: utf-8 -*-
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plate_gate_group import PLATE_GATE_GROUP_FIELDS, PlateGateGroup
from app.models.plate_white_list_settings import PlateWhiteListSettings


class PlateGateGroupRepository:
    @staticmethod
    async def list_all(db: AsyncSession):
        result = await db.execute(select(PlateGateGroup).order_by(PlateGateGroup.name))
        return result.scalars().all()

    @staticmethod
    async def get_by_id(db: AsyncSession, group_id: int) -> Optional[PlateGateGroup]:
        result = await db.execute(
            select(PlateGateGroup).where(PlateGateGroup.id == group_id)
        )
        return result.scalars().first()

    @staticmethod
    async def get_by_name(db: AsyncSession, name: str) -> Optional[PlateGateGroup]:
        """So tên KHÔNG phân biệt hoa thường: 'Làn A' và 'làn a' là một cụm.

        Hai cụm chỉ khác nhau cái viết hoa thì người dùng không phân biệt được
        mình đang gán camera vào cụm nào — đúng cái lỗi mà cụm sinh ra để chặn.
        """
        result = await db.execute(
            select(PlateGateGroup).where(
                PlateGateGroup.name.ilike(name)
            )
        )
        return result.scalars().first()

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> PlateGateGroup:
        entry = PlateGateGroup()
        for field in PLATE_GATE_GROUP_FIELDS:
            if data.get(field) is not None:
                setattr(entry, field, data[field])
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def update(db: AsyncSession, entry: PlateGateGroup, data: dict) -> PlateGateGroup:
        for field in PLATE_GATE_GROUP_FIELDS:
            if data.get(field) is not None:
                setattr(entry, field, data[field])
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def set_members(db: AsyncSession, group_id: int, camera_ids) -> None:
        """Đặt DANH SÁCH ĐẦY ĐỦ camera của cụm (gỡ những cái không còn trong list).

        Chỉ đụng tới camera ĐÃ bật whitelist: bảng này không tạo dòng mới, vì
        có dòng ở đây nghĩa là barrier đã bật cho camera đó — gán vào cụm mà
        vô tình bật barrier cho một camera là chuyện không được phép xảy ra.
        """
        wanted = {str(c) for c in (camera_ids or [])}
        # Gỡ camera cũ không còn trong danh sách.
        await db.execute(
            update(PlateWhiteListSettings)
            .where(
                PlateWhiteListSettings.gate_group_id == group_id,
                PlateWhiteListSettings.camera_id.notin_(wanted) if wanted else True,
            )
            .values(gate_group_id=None)
        )
        if wanted:
            await db.execute(
                update(PlateWhiteListSettings)
                .where(PlateWhiteListSettings.camera_id.in_(wanted))
                .values(gate_group_id=group_id)
            )
        await db.commit()

    @staticmethod
    async def delete(db: AsyncSession, entry: PlateGateGroup) -> None:
        # Gỡ camera khỏi cụm TRƯỚC khi xoá cụm. Bỏ qua bước này thì các camera
        # còn trỏ tới một id không còn tồn tại: chúng vẫn chạy (cache coi như
        # đứng riêng) nhưng DB mang dữ liệu rác không ai dọn.
        await db.execute(
            update(PlateWhiteListSettings)
            .where(PlateWhiteListSettings.gate_group_id == entry.id)
            .values(gate_group_id=None)
        )
        await db.delete(entry)
        await db.commit()
