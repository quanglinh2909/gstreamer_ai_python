# -*- coding: utf-8 -*-
"""Cột dùng chung của MỌI sự kiện AI.

Bốn bảng sự kiện (khuôn mặt, biển số, vùng cấm, khẩu trang) đã từng khai báo
lặp y hệt nhau bảy cột: camera_id, confidence, timestamp, image_full,
image_crop và bốn số của khung phát hiện. Lặp thì không sai ngay, nhưng mỗi
lần thêm một cột chung (như `box_*` lần trước) là phải sửa đúng bốn chỗ và chỉ
cần quên một chỗ là bảng đó lệch schema trong im lặng.

Mixin này giữ đúng phần CHUNG. Cái riêng của từng loại — identity_id của khuôn
mặt, plate_number của biển số, mask_status của khẩu trang — vẫn khai ở lớp con.
Khoá ngoại KHÔNG đặt được trong mixin thường (SQLAlchemy đòi `declared_attr`
vì mỗi bảng cần một đối tượng Column riêng), nên chúng nằm ở lớp con là đúng
chỗ chứ không phải thiếu sót.

Đường dẫn ảnh lưu dạng URL "/uploads/<...>" — cùng dạng mà
`AIServiceBase._save_images_blocking` trả về và `storage_cleanup_service` đọc
để xoá file.
"""

from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column


class AiEventMixin:
    """Bảy cột mà bảng sự kiện AI nào cũng có."""

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False)
    # Epoch GIÂY. Cũng là cột sắp xếp của mọi danh sách sự kiện và là cột
    # "cũ nhất trước" mà bộ dọn dung lượng dùng.
    timestamp: Mapped[int] = mapped_column(nullable=False)
    image_full: Mapped[str] = mapped_column(String, nullable=True)
    image_crop: Mapped[str] = mapped_column(String, nullable=True)
    # Khung phát hiện CHUẨN HOÁ [0,1] theo ảnh full (để vẽ box lên ảnh ở bất kỳ
    # kích thước hiển thị nào).
    box_x1: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_y1: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_x2: Mapped[Optional[float]] = mapped_column(nullable=True)
    box_y2: Mapped[Optional[float]] = mapped_column(nullable=True)

    def box_dict(self) -> Optional[dict]:
        """Khung dạng dict cho payload WebSocket/API, hoặc None nếu chưa có."""
        if self.box_x1 is None or self.box_y1 is None:
            return None
        return {
            "x1": self.box_x1, "y1": self.box_y1,
            "x2": self.box_x2, "y2": self.box_y2,
        }
