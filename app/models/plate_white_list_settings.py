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


# Các cột ngưỡng (tách khỏi camera_id) — dùng chung cho DTO, repository và
# cache để thêm cột mới chỉ phải sửa một chỗ.
PLATE_WHITE_LIST_SETTING_FIELDS = (
    "pre_time",
    "max_edit_distance",
    "ocr_confidence",
    "min_plate_length",
    "barrier_duration",
)
