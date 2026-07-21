import asyncio
import re
import sys
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_plate import EventPlate
from app.repositories.event_plate_repository import EventPlateRepository
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.services.ai_service_base import AIServiceBase
from app.tasks.task_parking_lot import task_parking_lot
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

# PLATE_SPEC = AIJobSpec(
#     config_type=TypeConfigAiEnum.PLATE_RECOGNITION.value,
#     transform_data="align_plate",
#     name="plate recognition",
#     model_file_1="plate_number_seg.rknn",
#     model_file_2="rf_detf_ocr.rknn",
#     model_type_1="yolov8_seg",
#     model_type_2="rf_detect",
# )



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

class PlateRecognitionService(AIServiceBase):
    # (camera_id, tracker_id) -> timestamp of the last EventPlate written for
    # that tracker. _track_state stops duplicates while a car stays in the
    # zone; this guards the exit/re-enter case (the car briefly leaves the
    # zone or detection drops) where _track_state is cleared but the tracker
    # keeps its id, which would otherwise write a second row. Longer than the
    # face window because a parked car can linger at the boundary.
    _REENTER_COOLDOWN_S = 30

    def __init__(self):
        # (camera_id, tracker_id) -> _PENDING / _RESOLVED. Lives across
        # frames of the same tracker; popped on exited_zone.
        self._track_state: dict = {}
        self._last_saved: dict = {}

    @staticmethod
    def _clean_plate(text_plate: str) -> str:
        # Drop punctuation/symbols but keep accented Vietnamese chars (some
        # province codes have diacritics) and whitespace; length is then
        # measured on this normalised form.
        return re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', text_plate or "")

    async def plate_recognition(self, db: AsyncSession, req: PlateRecognitionDTO):
        extra_data = {
            "pre_time": req.pre_time if req.pre_time is not None else 10,
        }
        return await ai_job_service.upsert(db, req, PLATE_SPEC, extra_data=extra_data)

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

    # `_find_parent` / `_make_crop` / `_save_images_blocking` come from
    # AIServiceBase; only the crop geometry, upload folder and filename
    # suffix differ.
    EVENT_FOLDER = "plates"

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
    CROP_OUTPUT_W = 280
    CROP_OUTPUT_H = 120

    @classmethod
    def _stem_suffix(cls, text_plate):
        # Plate images are named by the recognised text, not a tracker id,
        # so strip anything that isn't filename-safe.
        return re.sub(r"[^A-Za-z0-9_-]", "", text_plate) or "unknown"

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
                           key, log_label, persist: bool = True, extra_data=None):
        """Read the plate text from the detection's OCR children. Returns
        True when the plate is confirmed (len >= _PLATE_MIN_LEN), False when
        we still need more frames. Also fires the whitelist/barrier task for
        any partial read above the looser whitelist threshold — independent
        of DB persistence.

        When `persist` is False the OCR read and whitelist/barrier still run
        (so the gate keeps working every frame) but no EventPlate row is
        written and the state is left untouched — used once the tracker is
        already RESOLVED to avoid duplicate saves."""
        text_plate = detect_plate_from_children(
            parent.get("children", []), secondary_conf,
        )
        if not text_plate:
            return False
        t = self._clean_plate(text_plate)
        # Partial reads still useful for the whitelist (gate open) — the
        # whitelist service itself rate-limits duplicate hits per plate.
        
        if len(t) >= _PLATE_WHITELIST_MIN_LEN:
            pre_time = int((extra_data or {}).get("pre_time") or 10)
            asyncio.create_task(
                plate_white_list_service.process_ai_result(text_plate, pre_time)
            )
        if len(t) < _PLATE_MIN_LEN:
            return False

        task_parking_lot.add_task({
                "task": "plate_recognition",
                "plate": t.upper(),
                "timestamp": timestamp,
                "camera_id": meta["cameraId"],
                "full_jpeg": full_jpeg,
            })
        # Already confirmed and saved on an earlier frame: keep reading and
        # firing the whitelist above, but don't write a duplicate EventPlate.
        if not persist:
            return True
        # Exit/re-enter dedup: the same tracker may briefly leave the zone
        # (or detection drops) which clears _track_state, then re-enter under
        # the same id. Skip the duplicate row but mark RESOLVED so in_the_area
        # stops retrying.
        last = self._last_saved.get(key)
        if last is not None and timestamp - last < self._REENTER_COOLDOWN_S:
            self._track_state[key] = _RESOLVED
            return True
        # Confirmed — flip state synchronously before launching the async
        # save so the very next frame's in_the_area sees _RESOLVED and
        # doesn't schedule a duplicate persist.
        self._last_saved[key] = timestamp
        self._track_state[key] = _RESOLVED
        print(f"{log_label} id={key[1]} plate={text_plate}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, text_plate, timestamp)
        )
        return True

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
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
            key, "entered_zone", extra_data=extra_data,
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        self._track_state.pop((str(meta["cameraId"]), int(id)), None)
        # Keep recent save stamps for the re-enter cooldown; drop aged-out
        # ones so the map can't grow without bound.
        cutoff = timestamp - self._REENTER_COOLDOWN_S
        self._last_saved = {k: t for k, t in self._last_saved.items() if t >= cutoff}
        print(f"Plate exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        key = (str(meta["cameraId"]), int(id))
        state = self._track_state.get(key)
        # Ignore trackers that never entered the zone.
        if state is None:
            return
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        # _PENDING -> still confirming, persist on success. _RESOLVED -> keep
        # reading/whitelisting every frame but don't save a duplicate row.
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "in_the_area", persist=(state == _PENDING), extra_data=extra_data,
        )



plate_recognition_service = PlateRecognitionService()
