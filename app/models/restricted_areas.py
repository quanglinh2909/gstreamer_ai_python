from app.core.database import Base
from app.models.event_base import AiEventMixin


class RestrictedArea(Base, AiEventMixin):
    """Một lần có đối tượng bước vào vùng cấm. Cột chung xem AiEventMixin.

    Loại này KHÔNG có cột riêng nào — chính là bằng chứng rằng phần chung đã
    tách đúng chỗ.
    """

    __tablename__ = "restricted_areas"
