from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plate_white_list_settings import (
    PLATE_WHITE_LIST_SETTING_FIELDS,
    PlateWhiteListSettings,
)


class PlateWhiteListSettingsRepository:
    @staticmethod
    async def get_by_camera(
        db: AsyncSession, camera_id: str,
    ) -> Optional[PlateWhiteListSettings]:
        result = await db.execute(
            select(PlateWhiteListSettings).where(
                PlateWhiteListSettings.camera_id == camera_id
            )
        )
        return result.scalars().first()

    @staticmethod
    async def list_all(db: AsyncSession):
        """Dùng lúc khởi động để nạp cache. Mỗi camera đúng một dòng nên bảng
        này chỉ lớn bằng số camera — select không giới hạn là đủ."""
        result = await db.execute(
            select(PlateWhiteListSettings).order_by(PlateWhiteListSettings.id)
        )
        return result.scalars().all()

    @staticmethod
    async def upsert(
        db: AsyncSession, camera_id: str, settings: dict,
    ) -> PlateWhiteListSettings:
        entry = await PlateWhiteListSettingsRepository.get_by_camera(db, camera_id)
        if entry is None:
            entry = PlateWhiteListSettings(camera_id=camera_id)
            db.add(entry)
        # Chỉ nhận khoá nằm trong PLATE_WHITE_LIST_SETTING_FIELDS để một
        # payload lạ không ghi đè được camera_id hay id.
        for field in PLATE_WHITE_LIST_SETTING_FIELDS:
            if field in settings and settings[field] is not None:
                setattr(entry, field, settings[field])
        await db.commit()
        await db.refresh(entry)
        return entry

    @staticmethod
    async def delete(db: AsyncSession, entry: PlateWhiteListSettings) -> None:
        await db.delete(entry)
        await db.commit()
