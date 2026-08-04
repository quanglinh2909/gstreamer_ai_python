from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.event_base import AiEventMixin


class EventPlate(Base, AiEventMixin):
    """Một lần đọc được biển số. Cột chung xem AiEventMixin."""

    __tablename__ = "event_plates"

    plate_number: Mapped[str] = mapped_column(String(255), nullable=False)
