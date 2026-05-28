from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PlateWhiteList(Base):
    __tablename__ = "plate_white_list"

    id: Mapped[int] = mapped_column(primary_key=True)
    plate_number: Mapped[str] = mapped_column(String(12), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

