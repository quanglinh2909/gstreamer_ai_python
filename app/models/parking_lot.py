from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ParkingLot(Base):
    """A parking lot links one face camera and one plate camera together so the
    two streams can be correlated (e.g. match a recognised face with the plate
    captured at the same gate). Cameras live in an external service, so we store
    their string ids rather than a foreign key."""

    __tablename__ = "parking_lot"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # unique: a camera belongs to at most one parking lot.
    face_camera_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    plate_camera_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
