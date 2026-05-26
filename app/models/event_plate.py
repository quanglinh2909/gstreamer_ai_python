from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EventPlate(Base):
    __tablename__ = "event_plates"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False)
    plate_number: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    timestamp: Mapped[int] = mapped_column(nullable=False)
    image_full: Mapped[str] = mapped_column(String, nullable=True)
    image_crop: Mapped[str] = mapped_column(String, nullable=True)
