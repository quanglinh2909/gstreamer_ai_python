from sqlalchemy import JSON, Boolean, String, text
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
    # Free-form per-config JSON, forwarded to the service hooks (entered_zone,
    # dwell_alert, exited_zone, in_the_area). Defaults to {} both at the ORM
    # layer (default=dict) and in the DB (server_default '{}').
    extra_data: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    # Ghi khung phát hiện xuống bảng detection_slice để XEM LẠI vẽ được
    # box/pose và tìm sự kiện theo vùng vẽ trên hình. MẶC ĐỊNH TẮT — bật là
    # ghi liên tục theo mỗi khung hình AI xử lý.
    save_detections: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
