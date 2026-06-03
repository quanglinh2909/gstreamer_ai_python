from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.parking_lot_event_repository import ParkingLotEventRepository


class ParkingLotEventService:
    async def list_paginated(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        name: Optional[str] = None,
        identity_id: Optional[int] = None,
        plate_number: Optional[str] = None,
    ):
        return await ParkingLotEventRepository.list_paginated(
            db, page, size, name, identity_id, plate_number,
        )


parking_lot_event_service = ParkingLotEventService()
