from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.event_base import AiEventMixin


class EventFace(Base, AiEventMixin):
    """Một lần nhận diện khuôn mặt. Cột chung xem AiEventMixin."""

    __tablename__ = "event_face"

    # Người được khớp, NULL nếu không khớp ai (khách lạ).
    identity_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("identity.id", ondelete="SET NULL"), nullable=True
    )
