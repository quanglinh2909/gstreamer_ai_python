import asyncio
import os
from typing import Optional

from app.utils.play_sound import play_sound
from sqlalchemy.ext.asyncio import AsyncSession
import time
from app.utils.push_envent_metadata import push_event_metadata
from app.dto.face_mask_dto import FaceMaskDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.restricted_areas import RestrictedArea
from app.repositories.restricted_area_repository import RestrictedAreaRepository
from app.services.ai_job_service import AIJobSpec, ai_job_service
from app.utils.open_door.door_manager import door_manager


FACE_MASK_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.FACE_MASK.value,
    transform_data=None,
    name=TypeConfigAiEnum.FACE_MASK.value,
    model_file_1="yolov8m-mask.rknn",
    model_file_2=None,
    model_type_1="yolov8_detect",
    model_type_2=None,
    class_filter="0,3,5"
)

UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class FaceMaskService:
    TRACK_CLASS_IDS = frozenset({0})


    def __init__(self):
        self._hunmain: dict = {}

    async def face_mask(self, db: AsyncSession, req: FaceMaskDTO):
        extra_data = {
            "count_confirm": req.count_confirm if req.count_confirm is not None else 3,
            "re_alert_seconds": req.re_alert_seconds if req.re_alert_seconds is not None else 0,
        }
        return await ai_job_service.upsert(db, req, FACE_MASK_SPEC, extra_data=extra_data)

    # ─── Detection event hooks (driven by process_ai_service) ────────
    @staticmethod
    def _find_parent(meta, tid):
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None
    
    def calculate_containment(self, person_box, face_box):
        """Tính tỷ lệ face nằm trong person (0-1)"""
        px1, py1, px2, py2 = person_box
        fx1, fy1, fx2, fy2 = face_box

        # Tính diện tích giao nhau
        inter_x1 = max(px1, fx1)
        inter_y1 = max(py1, fy1)
        inter_x2 = min(px2, fx2)
        inter_y2 = min(py2, fy2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        face_area = (fx2 - fx1) * (fy2 - fy1)

        # Tỷ lệ face nằm trong person
        return inter_area / face_area if face_area > 0 else 0.0

    def _get_best_match_class_id(self, meta, parent):
        detections = meta.get("detections", [])
        x1 = parent.get("x1")
        y1 = parent.get("y1")
        x2 = parent.get("x2")
        y2 = parent.get("y2")
        
        person_box = (x1, y1, x2, y2)
        best_score = -1
        best_match_class_id = "UNKNOWN"
        for detection in detections:
            classId = detection.get("classId")
            confidence = detection.get("score", 0)
            if classId != 3 and classId != 5:  # Chỉ quan tâm đến face (classId 3) và mask (classId 5)
                continue
            face_box = (detection.get("x1"), detection.get("y1"), detection.get("x2"), detection.get("y2"))
            containment_ratio = self.calculate_containment(person_box, face_box)
            score = confidence * containment_ratio
            if containment_ratio >= 0.5 and score > best_score:
                best_score = score
                best_match_class_id = classId
        return best_match_class_id

    
    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id))
        x1 = parent.get("x1")
        y1 = parent.get("y1")
        x2 = parent.get("x2")
        y2 = parent.get("y2")
      
        best_match_class_id = self._get_best_match_class_id(meta, parent)

        if best_match_class_id == 5:
            play_sound.q_play_sound.put({"link": "access/mask.mp3", "time": timestamp})
            # push_event_metadata.push_event(
            #                         track_uuid=id,
            #                         timestamp=time.time(),
            #                         mask_status="face-mask",
            #                         bbox_x1=int(x1),
            #                         bbox_y1=int(y1),
            #                         bbox_x2=int(x2),
            #                         bbox_y2=int(y2),
            #                         image=full_jpeg
            #                     )
        else:
            play_sound.q_play_sound.put({"link": "access/welcome.mp3", "time": timestamp})
        
        self._hunmain[key] = {
            "timestamp": timestamp,
            "class_id_pre": best_match_class_id,
            "class_id_confirm":None,
            "count_confirm": 1 if best_match_class_id != "UNKNOWN" else 0
        }
        

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        print(f"restricted_area dwell_alert id={id}")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        key = (str(meta["cameraId"]), int(id))
        print(f"restricted_area exited_zone id={id}")

        if key in self._hunmain:
            del self._hunmain[key]

    # Emit the alert (sound + event) for a confirmed class. Shared by the
    # first-time confirmation and the periodic re-alert path so both behave
    # identically.
    def _fire_alert(self, id, class_id, meta, full_jpeg, x1, y1, x2, y2, timestamp):
        if class_id == 3:
            print(f"face_mask in_the_area id={id} - Face detected (no mask)")
            try:
                door_manager.open_door(0.5)
                print(f"Barrier opened ")
            except Exception as e:
                print(f"Failed to open barrier: {e}")


            push_event_metadata.push_event(
                track_uuid=id, timestamp=time.time(), mask_status="face",
                bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
                image=full_jpeg,
            )
        elif class_id == 5:
            print(f"face_mask in_the_area id={id} - Mask detected")
            play_sound.q_play_sound.put({"link": "access/warning.mp3", "time": timestamp})
            push_event_metadata.push_event(
                track_uuid=id, timestamp=time.time(), mask_status="face-mask",
                bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
                image=full_jpeg,
            )

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id))
        hunmain_entry = self._hunmain.get(key)
        if hunmain_entry is None:
            return

        # Per-config knobs (saved into AIConfig.extra_data via the API):
        #   count_confirm    – consecutive same-class frames before alerting.
        #   re_alert_seconds – if > 0, alert again every N seconds while the
        #                      same person keeps standing in the zone.
        extra = extra_data or {}
        confirm_threshold = int(extra.get("count_confirm") or 3)
        re_alert_seconds = int(extra.get("re_alert_seconds") or 0)

        best_match_class_id = self._get_best_match_class_id(meta, parent)

        if best_match_class_id == "UNKNOWN":
            return

        x1 = parent.get("x1")
        y1 = parent.get("y1")
        x2 = parent.get("x2")
        y2 = parent.get("y2")

        count_confirm = hunmain_entry.get("count_confirm", 0)
        class_id_pre = hunmain_entry.get("class_id_pre")
        if class_id_pre != best_match_class_id:
            # Class flipped (face <-> mask) — restart the confirmation count.
            hunmain_entry["count_confirm"] = 1
            hunmain_entry["class_id_pre"] = best_match_class_id
            return

        hunmain_entry["count_confirm"] = count_confirm + 1

        if hunmain_entry.get("class_id_confirm") != best_match_class_id:
            # Not yet confirmed for this class — fire once the streak is long
            # enough, then remember when we alerted.
            if count_confirm + 1 >= confirm_threshold:
                hunmain_entry["class_id_confirm"] = best_match_class_id
                hunmain_entry["last_alert_ts"] = timestamp
                self._fire_alert(id, best_match_class_id, meta, full_jpeg,
                                 x1, y1, x2, y2, timestamp)
        elif re_alert_seconds > 0:
            # Already confirmed and still standing there — re-alert on interval.
            last_alert_ts = hunmain_entry.get("last_alert_ts", timestamp)
            if timestamp - last_alert_ts >= re_alert_seconds:
                hunmain_entry["last_alert_ts"] = timestamp
                self._fire_alert(id, best_match_class_id, meta, full_jpeg,
                                 x1, y1, x2, y2, timestamp)
        
        


face_mask_service = FaceMaskService()
