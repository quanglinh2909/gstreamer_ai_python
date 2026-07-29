from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EventFace(Base):
    __tablename__ = "event_face"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False)
    identity_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("identity.id", ondelete="SET NULL"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(nullable=False)
    timestamp: Mapped[int] = mapped_column(nullable=False)
    image_full: Mapped[str] = mapped_column(String, nullable=True)
    image_crop: Mapped[str] = mapped_column(String, nullable=True)
    # Khung phát hiện CHUẨN HOÁ [0,1] theo ảnh full (để vẽ box lên ảnh).
    box_x1: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_y1: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_x2: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_y2: Mapped[Optional[float]] = mapped_column(nullable=True)
