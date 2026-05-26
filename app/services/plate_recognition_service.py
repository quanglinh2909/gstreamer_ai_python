import asyncio
import datetime
import os
import re
import sys

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_plate import EventPlate
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.utils.plate_recognition_hepper import detect_plate_from_children

PLATE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
    transform_data="align_plate",
    name="plate recognition",
    model_file_1="plate_number_seg.rknn",
    model_file_2="ocr.rknn",
    model_type_1="yolov8_seg",
    model_type_2="yolov8_detect",
)

# Project-root /uploads — same directory mounted as static in main.py.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class PlateRecognitionService:
    async def plate_recognition(self, db: AsyncSession, req: PlateRecognitionDTO):
        return await ai_job_service.upsert(db, req, PLATE_SPEC)

    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    @staticmethod
    def _save_images_blocking(full_jpeg, meta, parent, text_plate):
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
        folder_rel = os.path.join("plates", str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        safe_plate = re.sub(r"[^A-Za-z0-9_-]", "", text_plate) or "unknown"
        stem = f"{int(meta['seq']):010d}_{safe_plate}"
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

    async def _persist_event(self, meta, parent, full_jpeg, text_plate, timestamp):
        # Lazy import avoids the circular dep at module load; by the time this
        # task runs, process_ai_service._session_factory has been created on
        # the same event loop _persist_event is running on (the recv-loop's).
        from app.services.process_ai_service import process_ai_service
        session_factory = process_ai_service._session_factory
        if session_factory is None:
            return
        try:
            paths = await asyncio.to_thread(
                self._save_images_blocking, full_jpeg, meta, parent, text_plate,
            )
            if paths is None:
                return
            full_url, crop_url = paths
            async with session_factory() as db:
                db.add(
                    EventPlate(
                        camera_id=str(meta["cameraId"]),
                        type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
                        plate_number=text_plate,
                        confidence=float(parent.get("score", 0.0)),
                        timestamp=int(timestamp),
                        image_full=full_url,
                        image_crop=crop_url,
                    )
                )
                await db.commit()
        except Exception as exc:
            print(f"plate persist error: {exc}", file=sys.stderr)

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        text_plate = detect_plate_from_children(parent.get("children", []),secondary_conf)
        if not text_plate:
            return
        print(f"entered_zone id={id} plate={text_plate}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, text_plate, timestamp)
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf):
        # Implement logic to handle when a plate stayed in a zone
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        # Implement logic to handle when a plate exited a zone
        print(f"Plate exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf):
        # Implement logic to handle when a plate is in the area
        pass
        # print(f"in_the_area")
        # parent = self._find_parent(meta, id)
        # if parent is None:
        #     return
        # text_plate = detect_plate_from_children(parent.get("children", []))
        # print(f"entered_zone id={id} plate={text_plate}")



plate_recognition_service = PlateRecognitionService()
