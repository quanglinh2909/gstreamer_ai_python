from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.core.database import AsyncSessionLocal
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.repositories.ai_config_repository import AIRepository

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
        # Drop every cached job for this camera in the recv loop so any
        # in-flight tracker state is rebuilt (or stays empty if the camera
        # is really gone).
        from app.services.process_ai_service import process_ai_service
        process_ai_service.invalidate(camera_id)
        return data

    async def get_by_camera_id(self, db: AsyncSession, camera_id: str):
        ai_jobs = await HTTPXClient.get(f"/cameras/{camera_id}/ai-jobs")
        configs = await AIRepository.get_by_camera_id(db, camera_id)
        return [self._merge_ai_job(job, configs) for job in ai_jobs]

    def _merge_ai_job(self, ai_job: dict, configs) -> dict:
        # Match by job_id (saved into AIConfig on upsert) instead of by
        # transformData. The old transform-based lookup only worked for
        # stage-2 cascades — single-stage jobs like restricted_area have
        # no transformData and silently fell out, returning without
        # type/polygons/thresholds. job_id is the canonical link.
        job_id = ai_job.get("id")
        matched = next((c for c in configs if c.job_id == job_id), None)
        transform = ai_job.get("transformData")

        if matched:
            # Authoritative source: take everything we have a column for.
            ai_job["type"] = matched.type
            ai_job["polygons"] = matched.polygons or "[]"
            ai_job["primaryConf"] = matched.primary_conf
            ai_job["secondaryConf"] = matched.secondary_conf
            ai_job["tracker"] = matched.tracker
            ai_job["overlap_threshold"] = matched.overlap_threshold
            ai_job["dwellSeconds"] = matched.dwell_seconds or 0
            ai_job["maxFps"] = matched.fps
            ai_job["job_id"] = matched.job_id
            ai_job["extra_data"] = matched.extra_data or {}
        else:
            # Fallback for jobs that exist in the C++ engine but never
            # went through the Python upsert (no AIConfig row yet) —
            # at least give the UI a usable `type` label.
            ai_job["type"] = _TRANSFORM_TO_TYPE.get(transform, transform)
            ai_job["polygons"] = "[]"

        for field in _AI_JOB_FIELDS_TO_STRIP:
            ai_job.pop(field, None)
        return ai_job

    async def get_ai_config(self, camera_id: str, job_id: str):
        async with AsyncSessionLocal() as db:
            return await AIRepository.get_by_camera_and_job(db, camera_id, job_id)




camera_service = CameraService()
