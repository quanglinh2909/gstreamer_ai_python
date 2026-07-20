from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_config import AIConfig


class AIRepository:

    @staticmethod
    async def get_job_type_map(db: AsyncSession) -> dict:
        """Map every saved C++ job_id to its AI type.

        job_id is the canonical link between a C++ engine ai-job and the
        Python-side type (same join camera_service uses), so callers can label
        an engine job by type without relying on transformData/name.
        """
        result = await db.execute(select(AIConfig.job_id, AIConfig.type))
        return {row[0]: row[1] for row in result.all()}

    @staticmethod
    async def create_or_update(db: AsyncSession, payload: AIConfig):
        result = await db.execute(
            select(AIConfig).where(
                AIConfig.camera_id == payload.camera_id,
                AIConfig.type == payload.type
            )
        )
        ai_config = result.scalars().first()
        if ai_config:
            ai_config.polygons = payload.polygons
            ai_config.job_id = payload.job_id
            ai_config.primary_conf = payload.primary_conf
            ai_config.secondary_conf = payload.secondary_conf
            ai_config.fps = payload.fps
            ai_config.tracker = payload.tracker
            ai_config.overlap_threshold = payload.overlap_threshold
            ai_config.dwell_seconds = payload.dwell_seconds
            if payload.extra_data is not None:
                ai_config.extra_data = payload.extra_data
        else:
            ai_config = AIConfig(
                camera_id=payload.camera_id,
                type=payload.type,
                polygons=payload.polygons,
                job_id=payload.job_id,
                secondary_conf=payload.secondary_conf,
                primary_conf=payload.primary_conf,
                fps=payload.fps,
                tracker=payload.tracker,
                overlap_threshold=payload.overlap_threshold,
                dwell_seconds=payload.dwell_seconds,
                extra_data=payload.extra_data if payload.extra_data is not None else {},
            )
            db.add(ai_config)
        await db.commit()
        await db.refresh(ai_config)
        return ai_config

    @staticmethod
    async def delete_by_camera_id(db: AsyncSession, camera_id: str):
        result = await db.execute(
            select(AIConfig).where(AIConfig.camera_id == camera_id)
        )
        ai_configs = result.scalars().all()
        for ai_config in ai_configs:
            await db.delete(ai_config)
        await db.commit()

    @staticmethod
    async def get_by_camera_id(db: AsyncSession, camera_id: str):
        result = await db.execute(
            select(AIConfig).where(AIConfig.camera_id == camera_id)
        )
        ai_configs = result.scalars().all()
        return ai_configs

    @staticmethod
    async def get_by_camera_and_job(db: AsyncSession, camera_id: str, job_id: str):
        result = await db.execute(
            select(AIConfig).where(
                AIConfig.camera_id == camera_id,
                AIConfig.job_id == job_id,
            )
        )
        return result.scalars().first()
