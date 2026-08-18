"""Cách đọc biển 2: bám theo XE, biển số đi theo xe."""

from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.ai_job_service import AIJobSpec, AIStage, AIVariant
from app.services.plate_recognition.base import PlateRecognitionBase

# Model xe: lớp 5 là BIỂN SỐ, các lớp còn lại là loại xe.
VEHICLE_PLATE_CLASS = 5
VEHICLE_CLASSES = frozenset({0, 1, 2, 3, 4, 6, 7})

VEHICLE_PLATE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
    name="plate recognition",
    stages=(
        # Tầng 0: một model tìm CẢ xe lẫn biển số trên khung. Không transform
        # (tầng 0 luôn ăn cả khung hình).
        AIStage(model_file="vehicle.rknn", model_type="yolov8_detect"),
        # Tầng 1: chỉ chạy trên hộp biển số (lớp 5), khoanh DÒNG chữ trong
        # biển. Cắt thẳng theo hộp — align_plate cần mask của model phân vùng,
        # mà model xe là detect thuần nên không có gì để nắn.
        AIStage(model_file="plate_det.rknn", model_type="paddle_ocr_det",
                input_classes=str(VEHICLE_PLATE_CLASS)),
        # Tầng 2: đọc ký tự trong từng dòng chữ mà tầng 1 khoanh được.
        AIStage(model_file="plate_rec.rknn", model_type="paddle_ocr_rec"),
    ),
)

# Chỉ các lớp XE vào tracker; biển (lớp 5) là THUỘC TÍNH của xe trong khung
# hình đó, gắn vào xe bằng độ nằm-trong. Đúng khuôn của khẩu trang: track
# người, gắn trạng thái khẩu trang vào người.
VEHICLE_PPOCR_VARIANT = AIVariant(
    id="vehicle_ppocr",
    label="Bám theo xe (vehicle + PP-OCR)",
    spec=VEHICLE_PLATE_SPEC,
    track_classes=VEHICLE_CLASSES,
    attach_classes=frozenset({VEHICLE_PLATE_CLASS}),
    class_meta={VEHICLE_PLATE_CLASS: {"name": "Plate", "color": "#FF9500"}},
)


class VehiclePlateService(PlateRecognitionBase):
    """Bám theo XE, biển số đi theo xe.

    Thân xe to và liên tục nên một lượt xe vào = một track = một sự kiện, còn
    biển đọc được ở khung nào thì gắn vào khung đó. `subject()` của lớp cơ sở
    tự trả về cái biển gắn vào xe (nhờ attach_classes ở trên), nên toàn bộ
    logic xác nhận / lưu / whitelist dùng lại y nguyên.
    """

    VARIANTS = (VEHICLE_PPOCR_VARIANT,)
