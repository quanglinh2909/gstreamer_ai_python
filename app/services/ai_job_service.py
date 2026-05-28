from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.models.ai_config import AIConfig
from app.repositories.ai_config_repository import AIRepository


@dataclass(frozen=True)
class AIJobSpec:
    config_type: str
    transform_data: Optional[str]
    name: str
    model_file_1: str
    model_file_2: Optional[str]
    model_type_1: str
    model_type_2: Optional[str]
    # Comma-separated YOLO class IDs to keep, e.g. "0,1" for person+bicycle.
    # Empty / None / "all" means keep every class. Matches C++ parseClassFilter
    # in src/ai/Config.hpp — the engine drops detections whose cls_id isn't in
    # the parsed set before any tracker / stage-2 work.
    class_filter: Optional[str] = None


class AIJobService:
    @staticmethod
    def _get_path(ai_models, file_name):
        return next(
            (m["path"] for m in ai_models if m["fileName"] == file_name),
            None,
        )

    @staticmethod
    def _find_existing(ai_jobs, spec):
        """Locate the C++-side job that already represents this SPEC for
        this camera so we PUT (update) instead of POSTing a duplicate.

        Match by `name` first — it's unique per SPEC (face/plate/
        restricted_area each carry a distinct label) and works even for
        single-stage jobs whose `transformData` is empty. The C++ DTO
        serialises a missing transform as "" not null, so a naive
        `transformData == None` comparison always failed for
        restricted_area and silently created a new duplicate every save.

        Fall back to a transform match for backwards-compat with stage-2
        cascade jobs (face / plate) that were saved before the rename."""
        by_name = next(
            (j for j in ai_jobs if j.get("name") == spec.name), None,
        )
        if by_name is not None:
            return by_name
        if spec.transform_data:
            return next(
                (j for j in ai_jobs
                 if j.get("transformData") == spec.transform_data),
                None,
            )
        return None

    @staticmethod
    def _to_ratio(value: float) -> float:
        return value / 100.0 if value > 1 else value

    async def upsert(self, db: AsyncSession, req, spec: AIJobSpec):
        req.primaryConf = self._to_ratio(req.primaryConf)
        req.secondaryConf = self._to_ratio(req.secondaryConf)

        ai_jobs = await HTTPXClient.get(f"/cameras/{req.cameraId}/ai-jobs")
        existing = self._find_existing(ai_jobs, spec)

        if existing:
            payload = req.model_dump(exclude_none=True, exclude={"polygons"})
            payload["primaryConf"] = 0.2
            payload["secondaryConf"] = 0.2
            if spec.class_filter is not None:
                payload["classFilter"] = spec.class_filter
            data = await HTTPXClient.put(f"/ai-jobs/{existing['id']}", json=payload)
        else:
            ai_models = await HTTPXClient.get("/ai-models")
            payload = req.model_dump(exclude={"polygons"})
            payload["modelPath"] = self._get_path(ai_models, spec.model_file_1)
            payload["modelPath2"] = self._get_path(ai_models, spec.model_file_2)
            payload["modelType"] = spec.model_type_1
            payload["modelType2"] = spec.model_type_2
            payload["transformData"] = spec.transform_data
            payload["name"] = spec.name
            payload["primaryConf"] = 0.2
            payload["secondaryConf"] = 0.2
            # Pass through to the C++ engine. Empty string means "keep all
            # classes" (matches parseClassFilter); a CSV like "0,1" keeps
            # only those YOLO ids — drops everything else at the engine
            # before tracker / stage-2 work.
            payload["classFilter"] = spec.class_filter or ""
            data = await HTTPXClient.post("/ai-jobs", json=payload)

        await AIRepository.create_or_update(
            db,
            AIConfig(
                camera_id=req.cameraId,
                type=spec.config_type,
                polygons=req.polygons,
                job_id=data.get("id"),
                primary_conf=req.primaryConf,
                secondary_conf=req.secondaryConf,
                fps=req.maxFps,
                tracker=req.tracker,
                overlap_threshold=req.overlap_threshold,
                dwell_seconds=req.dwellSeconds,
            ),
        )
        # Force the recv loop to reload tracker/polygons/thresholds for this
        # (camera, job) instead of keeping the stale snapshot it cached on
        # the first inbound message. Lazy import — avoids the
        # ai_job_service ↔ process_ai_hepper ↔ face_recognition_service cycle.
        from app.services.process_ai_service import process_ai_service
        process_ai_service.invalidate(req.cameraId, data.get("id"))
        return data

    async def inference_model(
        self,
        image: tuple,
        model_path: str,
        model_type: str,
        model_path_2: str,
        model_type_2: str,
        transform_data: str,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        files = {"image": image}
        data = {
            "modelPath": model_path,
            "modelType": model_type,
            "modelPath2": model_path_2,
            "modelType2": model_type_2,
            "transformData": transform_data,
            "primaryConf": str(primary_conf),
            "secondaryConf": str(secondary_conf),
        }
        return await HTTPXClient.post("/inference/run", data=data, files=files)

    async def inference_with_spec(
        self,
        image: tuple,
        spec: AIJobSpec,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        ai_models = await HTTPXClient.get("/ai-models")
        return await self.inference_model(
            image=image,
            model_path=self._get_path(ai_models, spec.model_file_1),
            model_type=spec.model_type_1,
            model_path_2=self._get_path(ai_models, spec.model_file_2),
            model_type_2=spec.model_type_2,
            transform_data=spec.transform_data,
            primary_conf=primary_conf,
            secondary_conf=secondary_conf,
        )


ai_job_service = AIJobService()
