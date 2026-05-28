import asyncio
import datetime
import os
import re
import sys
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_plate import EventPlate
from app.repositories.event_plate_repository import EventPlateRepository
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.utils.image_crop import fixed_size_crop
from app.utils.plate_recognition_hepper import detect_plate_from_children
from app.services.plate_white_list_service import plate_white_list_service
from app.ws.plate_event_ws import plate_event_broadcaster

PLATE_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
    transform_data="align_plate",
    name="plate recognition",
    model_file_1="plate_number_seg.rknn",
    model_file_2="ocr.rknn",
    model_type_1="yolov8_seg",
    model_type_2="yolov8_detect",
)

# Per-tracker confirmation state. entered_zone seeds it, in_the_area drives
# the retry loop until a long-enough plate string is read, exited_zone
# clears it. Keyed by (camera_id, tracker_id).
_PENDING = "pending"     # haven't read a >=8-char plate yet; keep trying
_RESOLVED = "resolved"   # plate confirmed and event persisted; stop trying

# Filtered plate string must reach this many chars (letters+digits, spaces
# stripped) before we trust the OCR enough to write a DB row. Shorter
# reads are kept in PENDING — usually the tracker just caught the car at
# an angle that hides part of the plate, and a later frame will read it
# in full.
_PLATE_MIN_LEN = 8
# Whitelist / barrier path triggers at a looser threshold (partial reads
# are good enough to recognise a known plate); independent of DB save.
_PLATE_WHITELIST_MIN_LEN = 7

# Project-root /uploads — same directory mounted as static in main.py.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class PlateRecognitionService:
    def __init__(self):
        # (camera_id, tracker_id) -> _PENDING / _RESOLVED. Lives across
        # frames of the same tracker; popped on exited_zone.
        self._track_state: dict = {}

    @staticmethod
    def _clean_plate(text_plate: str) -> str:
        # Drop punctuation/symbols but keep accented Vietnamese chars (some
        # province codes have diacritics) and whitespace; length is then
        # measured on this normalised form.
        return re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', text_plate or "")

    async def plate_recognition(self, db: AsyncSession, req: PlateRecognitionDTO):
        return await ai_job_service.upsert(db, req, PLATE_SPEC)

    async def test_inference(
        self,
        image: tuple,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
    ):
        return await ai_job_service.inference_with_spec(
            image=image,
            spec=PLATE_SPEC,
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
        return await EventPlateRepository.list_paginated(db, page, size, camera_id)

    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    # Plate bbox from detection sits tight on the plate edges. Pad outward
    # before saving so the crop also shows a little of the surrounding car
    # body — easier for a human reviewer to locate the vehicle. More
    # padding on the sides ("rộng ra") because plates are wide-aspect.
    CROP_PAD_LEFT = 0.4
    CROP_PAD_RIGHT = 0.4
    CROP_PAD_TOP = 0.3
    CROP_PAD_BOTTOM = 0.3

    # Every saved plate crop comes out at exactly this size. 4:1 matches
    # the single-row Vietnamese car plate (470×110 mm = 4.27:1) so most
    # crops fill the frame cleanly; 2-row motorbike plates (~1.4:1) get
    # source-extended horizontally and centred — never stretched.
    CROP_OUTPUT_W = 480
    CROP_OUTPUT_H = 120
    CROP_PAD_COLOR = 114  # YOLO-style neutral grey for letterbox fill

    @classmethod
    def _make_plate_crop(cls, img, bx1, by1, bx2, by2):
        return fixed_size_crop(
            img, bbox=(bx1, by1, bx2, by2),
            pad_lrtb=(cls.CROP_PAD_LEFT, cls.CROP_PAD_RIGHT,
                      cls.CROP_PAD_TOP, cls.CROP_PAD_BOTTOM),
            output_size=(cls.CROP_OUTPUT_W, cls.CROP_OUTPUT_H),
            pad_color=cls.CROP_PAD_COLOR,
        )

    @classmethod
    def _save_images_blocking(cls, full_jpeg, meta, parent, text_plate):
        if not full_jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        crop = cls._make_plate_crop(
            img,
            float(parent["x1"]), float(parent["y1"]),
            float(parent["x2"]), float(parent["y2"]),
        )
        if crop is None:
            return None

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
        if not cv2.imwrite(crop_abs, crop):
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
                event = EventPlate(
                    camera_id=str(meta["cameraId"]),
                    plate_number=text_plate,
                    confidence=float(parent.get("score", 0.0)),
                    timestamp=int(timestamp),
                    image_full=full_url,
                    image_crop=crop_url,
                )
                db.add(event)
                await db.commit()
            # expire_on_commit=False on the session_factory, so event.id
            # stays populated after the commit without a refresh. Push
            # the same row to every WebSocket subscriber.
            plate_event_broadcaster.publish({
                "id": event.id,
                "camera_id": event.camera_id,
                "plate_number": event.plate_number,
                "whitelisted": plate_white_list_service.is_whitelisted(
                    event.plate_number
                ),
                "confidence": float(event.confidence),
                "timestamp": int(event.timestamp),
                "image_full": event.image_full,
                "image_crop": event.image_crop,
            })
        except Exception as exc:
            print(f"plate persist error: {exc}", file=sys.stderr)

    def _try_confirm_plate(self, meta, parent, full_jpeg, timestamp, secondary_conf,
                           key, log_label):
        """Read the plate text from the detection's OCR children. Returns
        True when the plate is confirmed (len >= _PLATE_MIN_LEN) and a
        persist task was scheduled, False when we still need more frames.
        Also fires the whitelist/barrier task for any partial read above
        the looser whitelist threshold — independent of DB persistence."""
        text_plate = detect_plate_from_children(
            parent.get("children", []), secondary_conf,
        )
        if not text_plate:
            return False
        t = self._clean_plate(text_plate)
        # Partial reads still useful for the whitelist (gate open) — the
        # whitelist service itself rate-limits duplicate hits per plate.
        if len(t) >= _PLATE_WHITELIST_MIN_LEN:
            asyncio.create_task(
                plate_white_list_service.process_ai_result(text_plate)
            )
        if len(t) < _PLATE_MIN_LEN:
            return False
        # Confirmed — flip state synchronously before launching the async
        # save so the very next frame's in_the_area sees _RESOLVED and
        # doesn't schedule a duplicate persist.
        self._track_state[key] = _RESOLVED
        print(f"{log_label} id={key[1]} plate={text_plate}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, text_plate, timestamp)
        )
        return True

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id))
        # Seed state to _PENDING so in_the_area takes over if this frame's
        # OCR was incomplete. _try_confirm_plate flips it to _RESOLVED
        # when it succeeds.
        self._track_state[key] = _PENDING
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "entered_zone",
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf):
        self._track_state.pop((str(meta["cameraId"]), int(id)), None)
        print(f"Plate exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf):
        key = (str(meta["cameraId"]), int(id))
        # Only retry while entered_zone (or a previous in_the_area attempt)
        # left the plate unconfirmed. Once _RESOLVED we never persist
        # again — avoids double-saves per car / per zone visit.
        if self._track_state.get(key) != _PENDING:
            return
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "in_the_area",
        )



plate_recognition_service = PlateRecognitionService()
