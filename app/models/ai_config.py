from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AIConfig(Base):
    __tablename__ = "ai_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(255), nullable=False)
    polygons: Mapped[str] = mapped_column(String, nullable=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    primary_conf: Mapped[float] = mapped_column(nullable=False)
    secondary_conf: Mapped[float] = mapped_column(nullable=False)
    fps: Mapped[int] = mapped_column(nullable=True)
    tracker: Mapped[str] = mapped_column(nullable=True)
    overlap_threshold: Mapped[float] = mapped_column(nullable=True)
    dwell_seconds: Mapped[int] = mapped_column(nullable=True, default=0)
