import asyncio
import datetime
import os
import sys
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.restricted_area_dto import RestrictedAreaDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.restricted_areas import RestrictedArea
from app.repositories.restricted_area_repository import RestrictedAreaRepository
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.utils.image_crop import fixed_size_crop
from app.ws.restricted_area_event_ws import restricted_area_event_broadcaster

# Single-stage detection job (no stage-2 / no alignment). class_filter
# "0,1" tells the C++ engine to drop every YOLO class except 0 and 1
# before tracking — keeps unrelated objects out of the restricted-area
# events entirely, no Python-side filter needed.
# RESTRICTED_AREA_SPEC = AIJobSpec(
#     config_type=TypeConfigAiEnum.RESTRICTED_AREA.value,
#     transform_data=None,
#     name="restricted area",
#     model_file_1="yolov8.rknn",
#     model_file_2=None,
#     model_type_1="yolov8_detect",
#     model_type_2=None,
#     class_filter="0",
# )

RESTRICTED_AREA_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.RESTRICTED_AREA.value,
    transform_data=None,
    name="restricted area",
    model_file_1="rf_detr_m.rknn",
    model_file_2=None,
    model_type_1="rf_detect",
    model_type_2=None,
    class_filter="1",
)

UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class RestrictedAreaService:
    async def restricted_area(self, db: AsyncSession, req: RestrictedAreaDTO):
        return await ai_job_service.upsert(db, req, RESTRICTED_AREA_SPEC)

    async def test_inference(
        self,
        image: tuple,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        return await ai_job_service.inference_with_spec(
            image=image,
            spec=RESTRICTED_AREA_SPEC,
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
        return await RestrictedAreaRepository.list_paginated(db, page, size, camera_id)

    # ─── Detection event hooks (driven by process_ai_service) ────────
    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    # YOLO bbox already wraps the whole object (whole person / bike /
    # whatever class_filter lets through), so the padding is modest —
    # just enough breathing room for a human reviewer to see context.
    CROP_PAD_LEFT = 0.2
    CROP_PAD_RIGHT = 0.2
    CROP_PAD_TOP = 0.2
    CROP_PAD_BOTTOM = 0.2

    # Every saved crop comes out at exactly this size. 2:3 portrait
    # matches the natural aspect of a standing person; non-person
    # classes (bike, etc.) get source-extended around the centre so the
    # final image is never stretched.
    CROP_OUTPUT_W = 400
    CROP_OUTPUT_H = 480
    CROP_PAD_COLOR = 114  # neutral grey for letterbox fill

    @classmethod
    def _make_crop(cls, img, bx1, by1, bx2, by2):
        return fixed_size_crop(
            img, bbox=(bx1, by1, bx2, by2),
            pad_lrtb=(cls.CROP_PAD_LEFT, cls.CROP_PAD_RIGHT,
                      cls.CROP_PAD_TOP, cls.CROP_PAD_BOTTOM),
            output_size=(cls.CROP_OUTPUT_W, cls.CROP_OUTPUT_H),
            pad_color=cls.CROP_PAD_COLOR,
        )

    @classmethod
    def _save_images_blocking(cls, full_jpeg, meta, parent, tid):
        if not full_jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        crop = cls._make_crop(
            img,
            float(parent.get("x1", 0.0)), float(parent.get("y1", 0.0)),
            float(parent.get("x2", 0.0)), float(parent.get("y2", 0.0)),
        )
        if crop is None:
            return None

        date = datetime.date.today().isoformat()
        folder_rel = os.path.join("restricted", str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        stem = f"{int(meta['seq']):010d}_{int(tid)}"
        full_abs = os.path.join(folder_abs, f"{stem}_full.jpg")
        crop_abs = os.path.join(folder_abs, f"{stem}_crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(full_jpeg)
        if not cv2.imwrite(crop_abs, crop):
            return None
        return (
            f"/uploads/{folder_rel}/{stem}_full.jpg",
            f"/uploads/{folder_rel}/{stem}_crop.jpg",
        )

    async def _persist_event(self, meta, parent, full_jpeg, tid, timestamp):
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
            async with session_factory() as db:
                event = RestrictedArea(
                    camera_id=str(meta["cameraId"]),
                    confidence=float(parent.get("score", 0.0)),
                    timestamp=int(timestamp),
                    image_full=full_url,
                    image_crop=crop_url,
                )
                db.add(event)
                await db.commit()
            # expire_on_commit=False on the session_factory keeps
            # event.id populated after commit without an extra refresh.
            # class_id isn't persisted yet (would need a schema bump),
            # but the detection's class id is already in `parent` so we
            # forward it on the WS frame — handy for the UI to render
            # different labels per YOLO class.
            restricted_area_event_broadcaster.publish({
                "id": event.id,
                "camera_id": event.camera_id,
                "class_id": parent.get("classId"),
                "confidence": float(event.confidence),
                "timestamp": int(event.timestamp),
                "image_full": event.image_full,
                "image_crop": event.image_crop,
            })
        except Exception as exc:
            print(f"restricted-area persist error: {exc}", file=sys.stderr)

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        print(f"restricted_area entered_zone id={id} class={parent.get('classId')}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, id, timestamp)
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf):
        print(f"restricted_area dwell_alert id={id}")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        print(f"restricted_area exited_zone id={id}")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf):
        # No re-persist while the tracker stays — entered_zone already
        # captured a row. dwell_alert handles "stayed too long".
        pass


restricted_area_service = RestrictedAreaService()
