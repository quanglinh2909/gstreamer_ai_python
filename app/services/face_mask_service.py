import asyncio
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.play_sound import play_sound
from app.utils.push_envent_metadata import push_event_metadata
from app.dto.face_mask_dto import FaceMaskDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.event_mask import EventMask
from app.repositories.event_mask_repository import EventMaskRepository
from app.services.ai_job_service import AIJobSpec, AIStage, AIVariant, ai_job_service
from app.services.ai_service_base import AIServiceBase
from app.utils.open_door.door_manager import door_manager
from app.ws.mask_event_ws import mask_event_broadcaster


FACE_MASK_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.FACE_MASK.value,
    name=TypeConfigAiEnum.FACE_MASK.value,
    stages=(
        AIStage(model_file="yolov8m-mask.rknn", model_type="yolov8_detect",
                class_filter="0,3,5"),
    ),
)

# Một model duy nhất tìm cả ba lớp: người (0), không khẩu trang (3), có khẩu
# trang (5). Chỉ NGƯỜI được track — hai lớp kia là TRẠNG THÁI của người đó
# trong khung hình này, gắn vào người bằng độ nằm-trong. Track chúng riêng thì
# id nhảy loạn mỗi lần người quay mặt đi.
#
# Loại này chỉ có một cách làm nên giao diện sẽ không hiện ô chọn.
FACE_MASK_VARIANT = AIVariant(
    id="yolov8_mask",
    label="YOLOv8 khẩu trang (một model)",
    spec=FACE_MASK_SPEC,
    track_classes=frozenset({0}),
    attach_classes=frozenset({3, 5}),
    # Debug-overlay metadata per classId (optional, cosmetic only — read by
    # the MJPEG overlay in app/utils/ai_debug_overlay.py):
    #   "name"  – hiển thị thay cho "cls=<id>" trong nhãn box.
    #   "color" – tô màu box, ghi đè màu xanh/đỏ theo zone. Nhận tuple BGR
    #             (kiểu OpenCV) hoặc chuỗi hex "#RRGGBB"/"RRGGBB" (RGB).
    # Class nào không khai báo thì giữ nguyên hành vi cũ (cls=<id>, màu zone).
    class_meta={
        0: {"name": "Person"},
        3: {"name": "No Mask", "color": "#34C759"},
        5: {"name": "Mask", "color": "#FF3B30"},
    },
)


class FaceMaskService(AIServiceBase):
    VARIANTS = (FACE_MASK_VARIANT,)

    def __init__(self):
        self._hunmain: dict = {}

    async def face_mask(self, db: AsyncSession, req: FaceMaskDTO):
        extra_data = {
            "count_confirm": req.count_confirm if req.count_confirm is not None else 3,
            "re_alert_seconds": req.re_alert_seconds if req.re_alert_seconds is not None else 0,
            "barrier_duration": (
                req.barrier_duration if req.barrier_duration is not None else 0.5
            ),
        }
        return await ai_job_service.upsert(db, req, FACE_MASK_SPEC, extra_data=extra_data)

    # ─── Detection event hooks (driven by process_ai_service) ────────
    #
    # Khẩu trang giờ lưu như ba loại AI kia: ảnh xuống /uploads/masks/, hàng
    # xuống bảng event_mask, WebSocket mang ĐƯỜNG DẪN thay vì base64. Trước đây
    # sự kiện chỉ bay qua WebSocket rồi mất — tải lại trang là trắng bảng, và
    # bộ dọn dung lượng không có gì để đếm.
    #
    # push_event_metadata vẫn được gọi, nhưng chỉ còn nuôi hàng đợi của luồng
    # MJPEG cho thiết bị ngoài (device_router) — nó cần BYTES thô, không dùng
    # được URL.
    EVENT_FOLDER = "masks"
    EVENT_MODEL = EventMask
    EVENT_BROADCASTER = mask_event_broadcaster
    EVENT_SOURCE = "face_mask"

    # Box của khẩu trang là box NGƯỜI (class 0) — cùng hình học với vùng cấm.
    CROP_PAD_LEFT = 0.2
    CROP_PAD_RIGHT = 0.2
    CROP_PAD_TOP = 0.2
    CROP_PAD_BOTTOM = 0.2
    CROP_OUTPUT_W = 400
    CROP_OUTPUT_H = 480

    # classId của model -> mask_status lưu vào DB và gửi lên giao diện.
    _MASK_STATUS = {3: "not_wearing_mask", 5: "wearing_mask"}

    async def list_events(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        camera_id: Optional[str] = None,
    ):
        return await EventMaskRepository.list_paginated(db, page, size, camera_id)

    def _get_best_match_class_id(self, meta, parent, extra_data=None):
        """Lớp trạng thái (3 = không khẩu trang / 5 = có) của người đang xét.

        Việc tìm box phụ nằm trong box được track là chuyện CHUNG của mọi loại
        AI (biển số gắn vào xe cũng y hệt) nên nằm ở AIServiceBase.find_attached
        và lấy danh sách lớp từ biến thể; ở đây chỉ còn dịch ra classId."""
        attached = self.find_attached(meta, parent, extra_data)
        return attached.get("classId") if attached else "UNKNOWN"

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id), int(zone_idx))

        best_match_class_id = self._get_best_match_class_id(meta, parent, extra_data)
        if best_match_class_id == 5:
            play_sound.q_play_sound.put({"link": "access/mask.mp3", "time": timestamp})
        else:
            play_sound.q_play_sound.put({"link": "access/welcome.mp3", "time": timestamp})

        self._hunmain[key] = {
            "timestamp": timestamp,
            "class_id_pre": best_match_class_id,
            # Bắt đầu ở trạng thái CHƯA confirm cho mọi class (kể cả mask) để
            # mask cũng đi qua count_confirm giống face -> lưu event nhất quán.
            "class_id_confirm": None,
            "count_confirm": 1 if best_match_class_id != "UNKNOWN" else 0,
            # Người vào đã đeo khẩu trang sẵn thì đã được chào bằng mask.mp3;
            # khi confirm lại class 5 thì lưu event nhưng KHÔNG phát warning.mp3.
            "greeted_mask_at_entry": best_match_class_id == 5,
        }
        

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"face_mask dwell_alert id={id}")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        key = (str(meta["cameraId"]), int(id), int(zone_idx))
        print(f"face_mask exited_zone id={id}")

        if key in self._hunmain:
            del self._hunmain[key]

    # Emit the alert (sound + event) for a confirmed class. Shared by the
    # first-time confirmation and the periodic re-alert path so both behave
    # identically.
    def _fire_alert(self, id, class_id, meta, full_jpeg, parent, timestamp,
                    is_save=True, alert_sound=True, barrier_duration=0.5,
                    confidence=0.0, extra_data=None):
        x1, y1 = parent.get("x1"), parent.get("y1")
        x2, y2 = parent.get("x2"), parent.get("y2")

        if class_id == 3:
            print(f"face_mask in_the_area id={id} - Face detected (no mask)")
            try:
                door_manager.open_door(barrier_duration)
                print(f"Barrier opened ")
            except Exception as e:
                print(f"Failed to open barrier: {e}")
            device_status = "face"
        elif class_id == 5:
            print(f"face_mask in_the_area id={id} - Mask detected")
            if alert_sound:
                play_sound.q_play_sound.put({"link": "access/warning.mp3", "time": timestamp})
            device_status = "face-mask"
        else:
            return

        if not is_save:
            return

        # Luồng MJPEG cho thiết bị ngoài: cần BYTES thô nên vẫn đi đường cũ.
        push_event_metadata.push_event(
            track_uuid=id, timestamp=time.time(), mask_status=device_status,
            bbox_x1=int(x1), bbox_y1=int(y1), bbox_x2=int(x2), bbox_y2=int(y2),
            image=full_jpeg,
            camera_id=str(meta["cameraId"]), confidence=float(confidence or 0.0),
        )
        # Ảnh + hàng DB + WebSocket + đánh thức ghi hình: AIServiceBase.
        asyncio.create_task(self.save_event(
            meta, parent, full_jpeg, id, timestamp,
            columns={
                "mask_status": self._MASK_STATUS.get(class_id, "unknown"),
                "track_id": int(id),
            },
            payload={
                "mask_status": self._MASK_STATUS.get(class_id, "unknown"),
                "track_id": int(id),
            },
            confidence=float(confidence or 0.0),
            extra_data=extra_data,
        ))

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id), int(zone_idx))
        hunmain_entry = self._hunmain.get(key)
        if hunmain_entry is None:
            return

        # Per-config knobs (saved into AIConfig.extra_data via the API):
        #   count_confirm    – consecutive same-class frames before alerting.
        #   re_alert_seconds – if > 0, alert again every N seconds while the
        #                      same person keeps standing in the zone.
        #   barrier_duration – độ dài xung mở barrier (giây), tuỳ phần cứng cổng.
        extra = extra_data or {}
        confirm_threshold = int(extra.get("count_confirm") or 3)
        re_alert_seconds = int(extra.get("re_alert_seconds") or 0)
        # `or 0.5` bắt cả None lẫn 0: xung 0 giây thì barrier không nhận được
        # tín hiệu, coi như cổng hỏng — rơi về mặc định an toàn hơn.
        barrier_duration = float(extra.get("barrier_duration") or 0.5)

        best_match_class_id = self._get_best_match_class_id(meta, parent, extra_data)

        if best_match_class_id == "UNKNOWN":
            return

        count_confirm = hunmain_entry.get("count_confirm", 0)
        class_id_pre = hunmain_entry.get("class_id_pre")
        if class_id_pre != best_match_class_id:
            # Class flipped (face <-> mask) — restart the confirmation count.
            hunmain_entry["count_confirm"] = 1
            hunmain_entry["class_id_pre"] = best_match_class_id
            return

        hunmain_entry["count_confirm"] = count_confirm + 1

        # Người vào đã đeo khẩu trang sẵn -> đã chào bằng mask.mp3 lúc vào,
        # nên khi confirm/re-alert class 5 thì tắt warning.mp3 (vẫn lưu event).
        alert_sound = not (
            best_match_class_id == 5 and hunmain_entry.get("greeted_mask_at_entry")
        )

        if hunmain_entry.get("class_id_confirm") != best_match_class_id:
            # Not yet confirmed for this class — fire once the streak is long
            # enough, then remember when we alerted.
            if count_confirm + 1 >= confirm_threshold:
                hunmain_entry["class_id_confirm"] = best_match_class_id
                hunmain_entry["last_alert_ts"] = timestamp
                self._fire_alert(id, best_match_class_id, meta, full_jpeg,
                                 parent, timestamp, is_save=True,
                                 alert_sound=alert_sound,
                                 barrier_duration=barrier_duration,
                                 confidence=secondary_conf,
                                 extra_data=extra_data)
        elif re_alert_seconds > 0:
            # Already confirmed and still standing there — re-alert on interval.
            last_alert_ts = hunmain_entry.get("last_alert_ts", timestamp)
            if timestamp - last_alert_ts >= re_alert_seconds:
                hunmain_entry["last_alert_ts"] = timestamp
                self._fire_alert(id, best_match_class_id, meta, full_jpeg,
                                 parent, timestamp, is_save=False,
                                 alert_sound=alert_sound,
                                 barrier_duration=barrier_duration,
                                 confidence=secondary_conf,
                                 extra_data=extra_data)
        
        


face_mask_service = FaceMaskService()
