from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Identity(Base):
    __tablename__ = "identity"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    image_full: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_crop: Mapped[Optional[str]] = mapped_column(String, nullable=True)
