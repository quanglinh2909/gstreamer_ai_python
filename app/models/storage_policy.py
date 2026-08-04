# -*- coding: utf-8 -*-
"""Chính sách tự dọn dung lượng.

Mô hình "GIỮ TỐI THIỂU N GB TRỐNG" (giống đầu ghi thật), KHÔNG phải "cấp ngân
sách cố định": cấp cố định thì tiến trình khác phình lên là đĩa vẫn đầy trong
khi chương trình vẫn ghi tới hạn mức của nó. Ở đây chương trình theo dõi CHỖ
TRỐNG THẬT của ổ — ai làm đầy cũng vậy — hễ trống tụt dưới `min_free_gb` thì tự
xoá dữ liệu CŨ NHẤT của chính nó (record + 4 loại event) cho tới khi trống lại
đạt `target_free_gb`. Nhờ vậy đĩa không bao giờ bị chương trình làm đầy.

Các trọng số `w_*` quyết định khi phải xoá thì GIỮ LẠI mỗi loại bao nhiêu phần
(loại trọng số cao được giữ nhiều hơn); tổng nên = 100. `identities` KHÔNG nằm
trong danh sách vì là dữ liệu gốc (người đã đăng ký), không phải sự kiện.
"""

from sqlalchemy import BigInteger, Boolean, Float, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class StoragePolicy(Base):
    __tablename__ = "storage_policy"

    # Bảng một-hàng: luôn dùng id = 1.
    id: Mapped[int] = mapped_column(primary_key=True)

    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    # Bắt đầu xoá khi chỗ trống < ngưỡng này (GB).
    min_free_gb: Mapped[float] = mapped_column(
        Float, nullable=False, default=10.0, server_default=text("10")
    )
    # Xoá cho tới khi chỗ trống ≥ ngưỡng này (GB) rồi dừng — biên trễ (hysteresis)
    # để khỏi xoá liên tục quanh mốc. Nên > min_free_gb vài GB.
    target_free_gb: Mapped[float] = mapped_column(
        Float, nullable=False, default=13.0, server_default=text("13")
    )

    # Trọng số giữ lại (%). Tổng nên = 100. Mặc định: record chiếm phần lớn.
    w_record: Mapped[float] = mapped_column(
        Float, nullable=False, default=56.0, server_default=text("56")
    )
    w_event_face: Mapped[float] = mapped_column(
        Float, nullable=False, default=11.0, server_default=text("11")
    )
    w_event_plate: Mapped[float] = mapped_column(
        Float, nullable=False, default=11.0, server_default=text("11")
    )
    w_parking_lot_event: Mapped[float] = mapped_column(
        Float, nullable=False, default=7.0, server_default=text("7")
    )
    w_restricted_area: Mapped[float] = mapped_column(
        Float, nullable=False, default=7.0, server_default=text("7")
    )
    # Hai loại vào sau. Ảnh của chúng nhỏ (một khung 640px, không có video),
    # nên phần được giữ ít hơn — nhưng vẫn là NÚM CHỈNH ĐƯỢC, khác hẳn bản
    # trước: chuyển động khi ấy dọn theo tuổi thọ của bản ghi, còn khẩu trang
    # thì chẳng dọn gì vì không có bảng nào.
    w_event_mask: Mapped[float] = mapped_column(
        Float, nullable=False, default=4.0, server_default=text("4")
    )
    w_motion_event: Mapped[float] = mapped_column(
        Float, nullable=False, default=4.0, server_default=text("4")
    )

    updated_at: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
