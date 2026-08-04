from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.event_base import AiEventMixin


class EventMask(Base, AiEventMixin):
    """Một lần xác nhận có/không đeo khẩu trang.

    Trước đây khẩu trang là loại DUY NHẤT không có bảng: sự kiện chỉ bay qua
    WebSocket kèm ảnh base64 rồi mất. Hệ quả là tải lại trang là trắng bảng, và
    bộ dọn dung lượng không có gì để đếm — nhìn đâu cũng thấy khẩu trang chiếm
    0 byte trong khi ảnh vẫn đi qua RAM của mọi client đang mở.

    Giờ nó lưu như ba loại kia: ảnh xuống /uploads/masks/<camera>/<ngày>/, hàng
    xuống bảng này, WebSocket chỉ còn mang ĐƯỜNG DẪN thay vì nhồi base64.
    """

    __tablename__ = "event_mask"

    # "wearing_mask" | "not_wearing_mask" | "unknown" — cùng bộ giá trị mà
    # WebSocket vẫn gửi, để giao diện không phải dịch hai lần.
    mask_status: Mapped[str] = mapped_column(String(32), nullable=False)
    # Id của TRACKER (người đang được bám), không phải id sự kiện. Giữ lại để
    # truy ngược một người qua nhiều lần báo trong cùng một lượt đứng.
    track_id: Mapped[Optional[int]] = mapped_column(nullable=True)
