"""Nhận dạng biển số — một loại AI, hai CÁCH LÀM.

Bố cục này là khuôn chung cho mọi loại AI có nhiều hơn một cách xử lý:

    <loại_ai>/
        base.py        — lớp cơ sở: tất cả những gì hai cách làm giống nhau
        <cách_1>.py    — cây model + biến thể + lớp xử lý riêng của cách 1
        <cách_2>.py    — ... cách 2
        __init__.py    — mặt tiền cho router + bảng tra biến thể -> lớp xử lý

Nguyên tắc: khác biệt nào diễn tả được bằng DỮ LIỆU thì để trong AIVariant
(cây model, track lớp nào, lớp phụ nào gắn vào vật được track) chứ đừng viết
thành mã rẽ nhánh; lớp con chỉ tồn tại để giữ biến thể của nó và là chỗ đặt
hành vi thật sự riêng khi nào phát sinh.

Loại AI chỉ có MỘT cách làm thì không cần thư mục — cứ một file như
face_recognition_service.py, khai đúng một AIVariant. Giao diện đọc số lượng
biến thể qua GET /ai-variants nên nó tự giấu ô chọn khi chỉ có một.
"""

from app.services.plate_recognition.base import PlateRecognitionBase
from app.services.plate_recognition.seg_ocr import (
    PLATE_SEG_OCR_VARIANT,
    PLATE_SPEC,
    PlateSegOcrService,
)
from app.services.plate_recognition.vehicle_ppocr import (
    VEHICLE_CLASSES,
    VEHICLE_PLATE_CLASS,
    VEHICLE_PLATE_SPEC,
    VEHICLE_PPOCR_VARIANT,
    VehiclePlateService,
)


class PlateRecognitionService(PlateRecognitionBase):
    """Mặt tiền của loại AI "nhận dạng biển số".

    Router gọi vào đây để lưu cấu hình / thử ảnh / liệt kê sự kiện, nên nó phải
    biết CẢ HAI biến thể để còn chọn theo `req.variant`. Còn lúc chạy thật thì
    mỗi camera đã chốt một biến thể, và `plate_handler` trao thẳng khung hình
    cho lớp xử lý tương ứng — không có hàm nào phải tự rẽ nhánh."""

    VARIANTS = (PLATE_SEG_OCR_VARIANT, VEHICLE_PPOCR_VARIANT)


plate_seg_ocr_service = PlateSegOcrService()
vehicle_plate_service = VehiclePlateService()
plate_recognition_service = PlateRecognitionService()

# Biến thể -> lớp xử lý khung hình của nó. Thêm cách đọc biển thứ ba chỉ là
# thêm một file, một lớp con và một dòng ở đây.
PLATE_HANDLERS = {
    PLATE_SEG_OCR_VARIANT.id: plate_seg_ocr_service,
    VEHICLE_PPOCR_VARIANT.id: vehicle_plate_service,
}


def plate_handler(extra_data=None):
    """Lớp xử lý ứng với biến thể camera đang chạy.

    Camera lưu từ trước khi có biến thể thứ hai thì extra_data chưa có khoá
    `variant` — rơi về bản cũ, đúng cái nó vẫn đang chạy."""
    return PLATE_HANDLERS.get(
        (extra_data or {}).get("variant"), plate_seg_ocr_service,
    )


__all__ = [
    "PLATE_HANDLERS",
    "PLATE_SEG_OCR_VARIANT",
    "PLATE_SPEC",
    "PlateRecognitionBase",
    "PlateRecognitionService",
    "PlateSegOcrService",
    "VEHICLE_CLASSES",
    "VEHICLE_PLATE_CLASS",
    "VEHICLE_PLATE_SPEC",
    "VEHICLE_PPOCR_VARIANT",
    "VehiclePlateService",
    "plate_handler",
    "plate_recognition_service",
    "plate_seg_ocr_service",
    "vehicle_plate_service",
]
