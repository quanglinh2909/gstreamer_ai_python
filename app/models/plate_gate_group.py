# -*- coding: utf-8 -*-
"""CỤM CỔNG: nhiều camera cùng điều khiển MỘT barrier.

Vì sao là bảng riêng chứ không phải một ô chữ trên từng camera: cụm có thời
gian chờ CỦA CHÍNH NÓ. Nếu cụm chỉ là cái nhãn, thì camera 1 chờ 30s còn
camera 2 chờ 20s sẽ không có câu trả lời đúng nào cho câu "xe vừa qua cụm này
thì chờ bao lâu" — lấy số nào cũng là đoán, và người dùng không nhìn thấy mình
đang bị lấy số nào. Cụm có bảng riêng thì con số đó chỉ có một, hiện rõ ngay
chỗ tạo cụm.

Camera KHÔNG thuộc cụm nào vẫn dùng `pre_time` của riêng nó như trước.
"""

from sqlalchemy import String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PlateGateGroup(Base):
    __tablename__ = "plate_gate_group"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Tên hiển thị, phải là duy nhất — hai cụm trùng tên thì người dùng không
    # phân biệt được mình đang gán camera vào cụm nào.
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Giây tối thiểu giữa 2 lần mở cổng cho CÙNG một biển, tính chung cho MỌI
    # camera trong cụm. Đây là con số THAY THẾ `pre_time` của từng camera khi
    # camera đó thuộc cụm — không cộng dồn, không lấy max.
    # 0 = mỗi biển chỉ mở được đúng một lần (giống ý nghĩa của pre_time).
    pre_time: Mapped[int] = mapped_column(
        nullable=False, server_default=text("0")
    )


PLATE_GATE_GROUP_FIELDS = ("name", "pre_time")
