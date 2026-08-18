"""Cách đọc biển 1: bám theo CHÍNH CÁI BIỂN."""

from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.ai_job_service import AIJobSpec, AIStage, AIVariant
from app.services.plate_recognition.base import PlateRecognitionBase

PLATE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
    name="plate recognition",
    stages=(
        # Tầng 0: khoanh vùng biển số trên cả khung.
        AIStage(model_file="plate_number_seg.rknn", model_type="yolov8_seg"),
        # Tầng 1: nắn phẳng biển rồi đọc từng ký tự.
        AIStage(model_file="ocr.rknn", model_type="yolov8_detect",
                transform="align_plate"),
    ),
)

# Không khai track_classes/attach_classes: mọi hộp đều được track và không có
# lớp phụ nào cần gắn — hộp được track CHÍNH LÀ cái biển.
PLATE_SEG_OCR_VARIANT = AIVariant(
    id="plate_seg_ocr",
    label="Bám theo biển số (seg + OCR)",
    spec=PLATE_SPEC,
)


class PlateSegOcrService(PlateRecognitionBase):
    """Bám theo BIỂN SỐ.

    Model phân vùng khoanh biển trên cả khung, tầng sau nắn phẳng biển
    (align_plate) rồi đọc từng ký tự. Vì hộp được track chính là cái biển nên
    `subject()` trả về luôn hộp đó — không có gì phải gắn vào đâu.

    Điểm yếu là chỗ này: biển chỉ vài chục pixel, xe nghiêng hay chói đèn là
    mất track, bắt lại là một tracker id mới và thành một sự kiện nữa cho cùng
    một chiếc xe. Bản `vehicle_ppocr` sinh ra để chữa đúng điểm đó.
    """

    VARIANTS = (PLATE_SEG_OCR_VARIANT,)
