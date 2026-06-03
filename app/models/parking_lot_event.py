from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ParkingLotEvent(Base):
    """One row per successful face<->plate correlation at a parking lot gate
    (i.e. every time TaskParkingLot.valid_success fires). Records who was
    matched, on which plate, and at which lot — independent of the raw
    EventFace / EventPlate rows so the gate-open history stays queryable."""

    __tablename__ = "parking_lot_event"

    id: Mapped[int] = mapped_column(primary_key=True)
    parking_lot_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("parking_lot.id", ondelete="SET NULL"), nullable=True
    )
    identity_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("identity.id", ondelete="SET NULL"), nullable=True
    )
    plate_number: Mapped[str] = mapped_column(String(32), nullable=False)
    face_camera_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    plate_camera_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # Full-frame snapshots captured at match time, one per paired camera.
    face_image_full: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    plate_image_full: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[int] = mapped_column(nullable=False)
