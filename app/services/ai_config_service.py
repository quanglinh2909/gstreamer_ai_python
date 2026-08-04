from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.repositories.ai_config_repository import AIRepository

# Fallback type for engine jobs with no AIConfig row (orphans), keyed by the
# stage-2 transform — same mapping camera_service uses.
_TRANSFORM_TO_TYPE = {
    "align_plate": TypeConfigAiEnum.PLATE_RECOGNITION.value,
    "align_face": TypeConfigAiEnum.FACE_RECOGNITION.value,
}


class AIConfigService:
    async def enabled_count(self, db: AsyncSession) -> dict:
        """Per-type count of AIs that are turned ON, plus the grand total.

        "Đang bật" is the C++ engine's `enabled` flag on each ai-job, not the
        mere existence of an AIConfig row — a configured AI can be toggled off
        in the engine. We pull every job from the engine, keep the enabled
        ones, and label each by type via its job_id (falling back to the
        transform mapping for orphan jobs with no AIConfig).
        """
        jobs = await HTTPXClient.get("/ai-jobs")
        job_type_map = await AIRepository.get_job_type_map(db)
        motion_cameras, motion_recording = await self._motion_counts()

        by_type = {t.value: 0 for t in TypeConfigAiEnum}
        for job in jobs or []:
            if not job.get("enabled"):
                continue
            job_type = job_type_map.get(job.get("id"))
            if job_type is None:
                # No AIConfig row — best-effort label from the transform.
                job_type = _TRANSFORM_TO_TYPE.get(job.get("transformData"))
            if job_type is None:
                continue  # truly unknown job — don't invent a bucket
            by_type[job_type] = by_type.get(job_type, 0) + 1

        return {
            "total": sum(by_type.values()),
            "by_type": by_type,
            "motion_cameras": motion_cameras,
            "motion_recording_cameras": motion_recording,
        }

    @staticmethod
    async def _motion_counts() -> tuple:
        """(số camera bật phát hiện chuyển động, trong đó bao nhiêu đang ghi
        theo chuyển động).

        Đọc từ engine chứ không từ DB của Python: cấu hình chuyển động thuộc về
        bảng cameras của engine, và engine là nơi duy nhất biết chắc camera nào
        đang thật sự chạy với cấu hình nào.

        Engine không trả lời được thì trả (0, 0) chứ không ném lỗi: cả thẻ
        thống kê AI sẽ mất chỉ vì một con số phụ là quá đắt.
        """
        try:
            cameras = await HTTPXClient.get("/cameras")
        except Exception:
            return 0, 0
        enabled = [c for c in (cameras or []) if c.get("motionEnabled")]
        recording = [
            c for c in enabled
            if c.get("recordingEnabled") and c.get("recordingMode") == "motion"
        ]
        return len(enabled), len(recording)


ai_config_service = AIConfigService()
