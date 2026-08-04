from typing import Optional

from sqlalchemy import String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PlateWhiteListSettings(Base):
    """Tham số mở barrier theo whitelist, cấu hình RIÊNG cho từng camera.

    Mỗi camera nhiều nhất một dòng (camera_id unique). Sự tồn tại của dòng
    này CHÍNH LÀ công tắc bật/tắt: camera không có dòng nào thì nhánh
    whitelist/barrier bị bỏ qua hoàn toàn, không có giá trị mặc định nào chạy
    thay. Mở barrier là hành động vật lý nên phải được bật rõ ràng cho từng
    camera, không tự chạy vì quên cấu hình.
    """

    __tablename__ = "plate_white_list_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )

    # Thời gian kế tiếp để cho phép mở cổng nếu biển này được đọc lại (từ lần
    # đọc trước) — dùng để hạn chế tần suất mở cổng. Nếu 0 thì không cho mở
    # cổng lại (một biển chỉ mở được đúng một lần cho tới khi service khởi
    # động lại hoặc dòng whitelist đó được sửa).
    pre_time: Mapped[int] = mapped_column(
        nullable=False, server_default=text("0")
    )

    # Số ký tự tối đa được phép sai giữa biển OCR đọc được và biển trong
    # whitelist — tức ngưỡng khớp mờ để chịu lỗi OCR. 0 = phải khớp tuyệt đối.
    # CÀNG CAO CÀNG DỄ DÃI: biển Việt Nam có nhiều biển chỉ khác nhau 1-2 ký
    # tự nên để 2 là hai xe khác nhau có thể mở nhầm cổng cho nhau.
    max_edit_distance: Mapped[int] = mapped_column(
        nullable=False, server_default=text("0")
    )

    # Ngưỡng tin cậy của TỪNG ký tự OCR khi dựng chuỗi biển cho nhánh
    # whitelist. Nhánh này đọc biển RIÊNG, không dùng chung secondaryConf của
    # AI job (vốn dành cho việc lưu EventPlate), nên chỉnh ngưỡng ở đây không
    # ảnh hưởng tới dữ liệu sự kiện.
    # Ký tự yếu hơn mức này bị LOẠI khỏi chuỗi — biển đọc ra sẽ ngắn đi và
    # thường rớt luôn ở min_plate_length. Để 0 là nhận cả những ký tự rác.
    ocr_confidence: Mapped[float] = mapped_column(
        nullable=False, server_default=text("0.3")
    )

    # Số ký tự tối thiểu (chữ + số, đã bỏ khoảng trắng và dấu gạch) của chuỗi
    # biển đọc được thì mới đem đi đối chiếu whitelist. Ngưỡng này dễ hơn
    # ngưỡng lưu EventPlate của AI job vì đọc thiếu một ký tự vẫn đủ để
    # nhận ra một biển đã đăng ký.
    min_plate_length: Mapped[int] = mapped_column(
        nullable=False, server_default=text("7")
    )

    # Độ dài xung mở barrier (giây) — tuỳ phần cứng từng cổng. Cùng ý nghĩa
    # với ParkingLot.barrier_duration nhưng là cột RIÊNG: một camera có thể
    # vừa nằm trong bãi xe vừa chạy whitelist, và hai luồng đó có thể điều
    # khiển hai barrier khác nhau.
    barrier_duration: Mapped[float] = mapped_column(
        nullable=False, server_default=text("0.5")
    )

    # CỤM CỔNG mà camera này thuộc về (plate_gate_group.id). NULL = đứng
    # riêng, dùng `pre_time` ở trên như trước.
    #
    # Vì sao cần: mặc định `pre_time` chỉ chặn chính camera vừa đọc, và đó là
    # CỐ Ý — cổng vào và cổng ra là hai camera khác nhau, xe vừa vào không
    # được vì thế mà bị khoá ở cổng ra. Nhưng một làn vừa vào vừa ra thì hai
    # camera cùng nhìn MỘT barrier: xe chạy qua camera 1 mở cổng, chạy tiếp
    # qua camera 2 lại mở lần nữa, xung mở nối nhau nên barrier không kịp
    # đóng. Xếp hai camera đó vào cùng một cụm thì chúng dùng chung một đồng
    # hồ chờ và lần đọc thứ hai bị chặn.
    #
    # Khi thuộc cụm, thời gian chờ lấy của CỤM (plate_gate_group.pre_time),
    # không phải `pre_time` của camera — xem plate_gate_group.py.
    #
    # Không đặt ForeignKey: cụm bị xoá thì router tự gỡ camera khỏi cụm, và
    # bảng này còn phải sống được trên máy chưa từng có bảng cụm.
    gate_group_id: Mapped[Optional[int]] = mapped_column(
        nullable=True, server_default=text("NULL")
    )


# Các cột ngưỡng (tách khỏi camera_id) — dùng chung cho DTO, repository và
# cache để thêm cột mới chỉ phải sửa một chỗ.
PLATE_WHITE_LIST_SETTING_FIELDS = (
    "pre_time",
    "max_edit_distance",
    "ocr_confidence",
    "min_plate_length",
    "barrier_duration",
    "gate_group_id",
)
