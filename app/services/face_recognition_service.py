import asyncio
import datetime
import os
import sys
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.face_recognition_dto import FaceRecognitionDTO
from app.dto.identity_dto import FaceInfo
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_face import EventFace
from app.repositories.event_face_repository import EventFaceRepository
from app.repositories.face_vector_repository import FaceVectorRepository
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

UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class FaceRecognitionService:
    async def face_recognition(self, db: AsyncSession, req: FaceRecognitionDTO):
        return await ai_job_service.upsert(db, req, FACE_SPEC)

    async def test_inference(
        self,
        image: tuple,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        return await ai_job_service.inference_with_spec(
            image=image,
            spec=FACE_SPEC,
            primary_conf=primary_conf,
            secondary_conf=secondary_conf,
        )

    async def list_events(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        return await EventFaceRepository.list_paginated(db, page, size, camera_id)

    async def register_face(self, identity_id: int, image: tuple) -> FaceInfo:
        result = await ai_job_service.inference_with_spec(
            image=image,
            spec=FACE_SPEC,
        )
        detections = (result or {}).get("detections") or []
        candidates = [(d, self._extract_embedding(d)) for d in detections]
        candidates = [(d, e) for d, e in candidates if e]
        if not candidates:
            raise HTTPException(
                status_code=400, detail="No face/embedding detected in image"
            )

        best_det, best_emb = max(candidates, key=lambda x: float(x[0].get("score", 0.0)))

        milvus_id = await asyncio.to_thread(
            FaceVectorRepository.insert,
            embedding=best_emb,
            identity_id=identity_id,
        )
        return FaceInfo(
            id=milvus_id,
            score=float(best_det.get("score", 0.0)),
            embedding=best_emb,
        )

    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    @staticmethod
    def _save_images_blocking(full_jpeg, meta, parent, tid):
        if not full_jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, min(int(parent["x1"]), w - 1))
        y1 = max(0, min(int(parent["y1"]), h - 1))
        x2 = max(x1 + 1, min(int(parent["x2"]), w))
        y2 = max(y1 + 1, min(int(parent["y2"]), h))

        date = datetime.date.today().isoformat()
        folder_rel = os.path.join("faces", str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        stem = f"{int(meta['seq']):010d}_{int(tid)}"
        full_abs = os.path.join(folder_abs, f"{stem}_full.jpg")
        crop_abs = os.path.join(folder_abs, f"{stem}_crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(full_jpeg)
        if not cv2.imwrite(crop_abs, img[y1:y2, x1:x2]):
            return None
        return (
            f"/uploads/{folder_rel}/{stem}_full.jpg",
            f"/uploads/{folder_rel}/{stem}_crop.jpg",
        )

    @staticmethod
    def _extract_embedding(parent):
        for key in ("embedding", "feature", "features"):
            vec = parent.get(key)
            if vec:
                return list(vec)
        return None

    @staticmethod
    async def _match_identity(embedding,secondary_conf):
        hits = await asyncio.to_thread(
            FaceVectorRepository.search, embedding=embedding, top_k=1,
        )
        if not hits:
            return None
        top = hits[0]
        if float(top.get("score", 0.0)) < secondary_conf:
            return None
        identity_id = top.get("identity_id")
        return int(identity_id) if identity_id else None

    async def _persist_event(self, meta, parent, full_jpeg, tid, timestamp,secondary_conf):
        from app.services.process_ai_service import process_ai_service
        session_factory = process_ai_service._session_factory
        if session_factory is None:
            return
        try:
            paths = await asyncio.to_thread(
                self._save_images_blocking, full_jpeg, meta, parent, tid,
            )
            if paths is None:
                return
            full_url, crop_url = paths

            identity_id = None
            embedding = self._extract_embedding(parent)
            if embedding:
                identity_id = await self._match_identity(embedding,secondary_conf)

            async with session_factory() as db:
                db.add(
                    EventFace(
                        camera_id=str(meta["cameraId"]),
                        identity_id=identity_id,
                        confidence=float(parent.get("score", 0.0)),
                        timestamp=int(timestamp),
                        image_full=full_url,
                        image_crop=crop_url,
                    )
                )
                await db.commit()
        except Exception as exc:
            print(f"face persist error: {exc}", file=sys.stderr)

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        print(f"entered_zone id={id}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, id, timestamp,secondary_conf)
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        print(f"Face exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf):
        pass


face_recognition_service = FaceRecognitionService()
