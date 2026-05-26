from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.core.database import AsyncSessionLocal
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.repositories.ai_config_repository import AIRepository
from app.utils.convert_data import convert_config_type

_TRANSFORM_TO_TYPE = {
    "align_plate": TypeConfigAiEnum.PLATE_RECOGNITION.value,
    "align_face": TypeConfigAiEnum.FACE_RECOGNITION.value,
}

_AI_JOB_FIELDS_TO_STRIP = (
    "name",
    "modelPath",
    "modelType",
    "classFilter",
    "modelPath2",
    "modelType2",
    "transformData",
)


class CameraService:
    async def delete_cammera(self, db: AsyncSession, camera_id: str):
        data = await HTTPXClient.delete(f"/cameras/{camera_id}")
        await AIRepository.delete_by_camera_id(db, camera_id)
        return data

    async def get_by_camera_id(self, db: AsyncSession, camera_id: str):
        ai_jobs = await HTTPXClient.get(f"/cameras/{camera_id}/ai-jobs")
        configs = await AIRepository.get_by_camera_id(db, camera_id)
        return [self._merge_ai_job(job, configs) for job in ai_jobs]

    def _merge_ai_job(self, ai_job: dict, configs) -> dict:
        transform = ai_job.get("transformData")
        ai_job["type"] = _TRANSFORM_TO_TYPE.get(transform, transform)
        ai_job["polygons"] = "[]"

        matched = next(
            (c for c in configs if convert_config_type(c.type) == transform),
            None,
        )
        if matched:
            ai_job["polygons"] = matched.polygons
            ai_job["primaryConf"] = matched.primary_conf
            ai_job["secondaryConf"] = matched.secondary_conf

        for field in _AI_JOB_FIELDS_TO_STRIP:
            ai_job.pop(field, None)
        return ai_job

    async def get_ai_config(self, camera_id: str, job_id: str):
        async with AsyncSessionLocal() as db:
            return await AIRepository.get_by_camera_and_job(db, camera_id, job_id)


camera_service = CameraService()
