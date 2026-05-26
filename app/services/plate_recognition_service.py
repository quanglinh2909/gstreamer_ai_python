from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.ai_job_service import AIJobSpec, ai_job_service

PLATE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
    transform_data="align_plate",
    name="plate recognition",
    model_file_1="plate_number_seg.rknn",
    model_file_2="ocr.rknn",
    model_type_1="yolov8_seg",
    model_type_2="yolov8_detect",
)


class PlateRecognitionService:
    async def plate_recognition(self, db: AsyncSession, req: PlateRecognitionDTO):
        return await ai_job_service.upsert(db, req, PLATE_SPEC)


plate_recognition_service = PlateRecognitionService()
