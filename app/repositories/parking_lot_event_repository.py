from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.identity import Identity
from app.models.parking_lot import ParkingLot
from app.models.parking_lot_event import ParkingLotEvent


class ParkingLotEventRepository:
    @staticmethod
    async def list_paginated(
        db: AsyncSession,
        page: int,
        size: int,
        name: Optional[str] = None,
        identity_id: Optional[int] = None,
        plate_number: Optional[str] = None,
    ):
        filters = []
        if identity_id is not None:
            filters.append(ParkingLotEvent.identity_id == identity_id)
        if plate_number:
            filters.append(ParkingLotEvent.plate_number.ilike(f"%{plate_number}%"))
        if name:
            # Substring match on the matched identity's name (case-insensitive).
            filters.append(Identity.name.ilike(f"%{name}%"))

        # The Identity outerjoin is always present so the `name` filter and the
        # count agree; outerjoin keeps events whose identity was deleted.
        total = await db.scalar(
            select(func.count())
            .select_from(ParkingLotEvent)
            .outerjoin(Identity, ParkingLotEvent.identity_id == Identity.id)
            .where(*filters)
        )

        result = await db.execute(
            select(ParkingLotEvent, Identity.name, ParkingLot.name)
            .outerjoin(Identity, ParkingLotEvent.identity_id == Identity.id)
            .outerjoin(ParkingLot, ParkingLotEvent.parking_lot_id == ParkingLot.id)
            .where(*filters)
            .order_by(ParkingLotEvent.timestamp.desc(), ParkingLotEvent.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )

        events = []
        for event, identity_name, lot_name in result.all():
            event.name = identity_name
            event.parking_lot_name = lot_name
            events.append(event)
        return events, int(total or 0)
