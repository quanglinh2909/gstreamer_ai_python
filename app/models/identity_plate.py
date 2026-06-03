from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class IdentityPlate(Base):
    __tablename__ = "identity_plate"

    id: Mapped[int] = mapped_column(primary_key=True)
    identity_id: Mapped[int] = mapped_column(
        ForeignKey("identity.id", ondelete="CASCADE"), nullable=False
    )
    plate_number: Mapped[str] = mapped_column(String(12), nullable=False)
