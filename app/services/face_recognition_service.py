from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.face_recognition_dto import FaceRecognitionDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.ai_job_service import AIJobSpec, ai_job_service

FACE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.FACE_RECOGNITION.value,
    transform_data="align_face",
    name="Face recognition",
    model_file_1="yolov8_pose_face_in8.rknn",
    model_file_2="adaface_ir101_fp16.rknn",
    model_type_1="yolov8_pose",
    model_type_2="face_recognition",
)


class FaceRecognitionService:
    async def face_recognition(self, db: AsyncSession, req: FaceRecognitionDTO):
        return await ai_job_service.upsert(db, req, FACE_SPEC)

    def entered_zone(self, id, meta, full_jpeg, timestamp):
        # Implement logic to handle when a plate enters a zone
        print(f"entered_zone")

    def dwell_alert(self, id, meta, full_jpeg, timestamp):
        # Implement logic to handle when a plate stayed in a zone
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp):
        # Implement logic to handle when a plate exited a zone
        print(f"Plate exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp):
        # Implement logic to handle when a plate is in the area
        print(f"in_the_area")


face_recognition_service = FaceRecognitionService()
