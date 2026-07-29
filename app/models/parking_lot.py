from typing import Optional

from sqlalchemy import String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ParkingLot(Base):
    """A parking lot links one face camera and one plate camera together so the
    two streams can be correlated (e.g. match a recognised face with the plate
    captured at the same gate). Cameras live in an external service, so we store
    their string ids rather than a foreign key.

    Các cột phía dưới trước đây là hằng số trong TaskParkingLot. Chúng thuộc về
    từng cổng chứ không phải toàn hệ thống: cổng xe máy và cổng ô tô khác nhau
    về khoảng cách giữa hai camera, tốc độ xe qua và cả phần cứng barrier.
    server_default giữ đúng giá trị đã fix cứng trước đây nên các bãi đang chạy
    không đổi hành vi sau khi thêm cột.
    """

    __tablename__ = "parking_lot"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # unique: a camera belongs to at most one parking lot.
    face_camera_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    plate_camera_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )

    # Cửa sổ ghép cặp mặt ↔ biển (giây). Một khuôn mặt đã nhận diện chờ tối đa
    # ngần này giây để biển số của chính người đó xuất hiện ở camera kia (và
    # ngược lại). Hai camera đặt càng xa nhau, xe đi càng chậm thì cần càng
    # dài; để quá dài thì xe sau dễ bị ghép nhầm với người của xe trước.
    time_expired: Mapped[int] = mapped_column(
        nullable=False, server_default=text("30")
    )

    # Một biển số ở làn này chỉ tạo MỘT sự kiện trong ngần này giây. Chặn ca 2
    # người ngồi cùng xe: cả hai khuôn mặt đều khớp cùng một biển, nếu không
    # chặn thì tạo 2 dòng và mở barrier 2 lần. Đặt xấp xỉ thời gian một lượt
    # xe rời khỏi làn.
    match_cooldown: Mapped[int] = mapped_column(
        nullable=False, server_default=text("30")
    )

    # Độ dài xung mở barrier (giây) — tuỳ phần cứng từng cổng.
    barrier_duration: Mapped[float] = mapped_column(
        nullable=False, server_default=text("0.5")
    )

    # Số ký tự tối đa được phép sai giữa biển OCR đọc được và biển đã đăng ký
    # của cư dân. 0 = khớp tuyệt đối. Lưu ý biển ngắn còn bị siết thêm bởi luật
    # an toàn len//4 trong identity_plate_service (biển 4 ký tự luôn phải khớp
    # tuyệt đối dù đặt bao nhiêu).
    max_edit_distance: Mapped[int] = mapped_column(
        nullable=False, server_default=text("2")
    )

    # Ngưỡng tin cậy của TỪNG ký tự OCR khi bãi xe đọc lại biển. Nhánh này đọc
    # biển RIÊNG chứ không dùng chuỗi mà nhánh lưu EventPlate đã dựng bằng
    # secondaryConf của AI job — chỉnh ngưỡng cho barrier không kéo theo thay
    # đổi dữ liệu sự kiện. Ký tự yếu hơn bị LOẠI khỏi chuỗi.
    ocr_confidence: Mapped[float] = mapped_column(
        nullable=False, server_default=text("0.3")
    )

    # Độ chính xác khuôn mặt: điểm cosine tối thiểu để một khuôn mặt được coi
    # là KHỚP cư dân rồi mới đẩy sang luồng ghép cặp bãi xe (mở barrier). Đây
    # là ngưỡng RIÊNG cho nhánh bãi xe — trước đây fix cứng 0.15 trong
    # face_recognition_service. Cao hơn = ít nhận nhầm người lạ thành cư dân
    # (an toàn hơn) nhưng dễ bỏ sót; thấp hơn thì ngược lại. Không đụng tới
    # ngưỡng lưu sự kiện khuôn mặt thường (secondaryConf của AI job).
    face_confidence: Mapped[float] = mapped_column(
        nullable=False, server_default=text("0.15")
    )


# Các cột cấu hình (tách khỏi name / cặp camera) — dùng chung cho DTO,
# repository và cache để thêm cột mới chỉ phải sửa một chỗ.
PARKING_LOT_SETTING_FIELDS = (
    "time_expired",
    "match_cooldown",
    "barrier_duration",
    "max_edit_distance",
    "ocr_confidence",
    "face_confidence",
)
