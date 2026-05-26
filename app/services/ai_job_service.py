from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.httpx_client import HTTPXClient
from app.models.ai_config import AIConfig
from app.repositories.ai_config_repository import AIRepository


@dataclass(frozen=True)
class AIJobSpec:
    config_type: str
    transform_data: str
    name: str
    model_file_1: str
    model_file_2: str
    model_type_1: str
    model_type_2: str


class AIJobService:
    @staticmethod
    def _get_path(ai_models, file_name):
        return next(
            (m["path"] for m in ai_models if m["fileName"] == file_name),
            None,
        )

    @staticmethod
    def _find_existing(ai_jobs, transform_data):
        return next(
            (j for j in ai_jobs if j["transformData"] == transform_data),
            None,
        )

    @staticmethod
    def _to_ratio(value: float) -> float:
        return value / 100.0 if value > 1 else value

    async def upsert(self, db: AsyncSession, req, spec: AIJobSpec):
        req.primaryConf = self._to_ratio(req.primaryConf)
        req.secondaryConf = self._to_ratio(req.secondaryConf)

        ai_jobs = await HTTPXClient.get(f"/cameras/{req.cameraId}/ai-jobs")
        existing = self._find_existing(ai_jobs, spec.transform_data)

        if existing:
            payload = req.model_dump(exclude_none=True, exclude={"polygons"})
            payload["primaryConf"] = 0.2
            payload["secondaryConf"] = 0.2
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
