"""Shared plumbing for the per-camera AI services.

Every AI service (restricted area, face recognition, plate recognition,
face mask) is driven by the same detection hooks from `process_ai_service`
and — when it persists an event — runs the same routine: find the
detection carrying a tracker id, decode the frame, cut a fixed-size crop,
then write the full frame + crop under
`/uploads/<folder>/<cameraId>/<date>/`.

Only three things actually differ between services: the crop geometry,
the upload subfolder, and the filename suffix. Those are class attributes
(and one small override) here instead of four near-identical copies of
the same code.

Lưu MỘT sự kiện cũng là một thủ tục giống hệt nhau ở cả bốn loại (ghi ảnh →
chèn hàng → bắn WebSocket → đánh thức ghi hình), nên nó nằm ở `save_event`
dưới đây; lớp con chỉ khai `EVENT_MODEL` / `EVENT_BROADCASTER` và đưa thêm
cột riêng của mình.
"""

import asyncio
import datetime
import os
import sys

import cv2
import numpy as np

from app.utils.image_crop import fixed_size_crop

# <repo>/uploads — services import this from here so every one of them
# resolves the same directory. Re-exported by the service modules that
# used to define it locally, so existing imports keep working.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class AIServiceBase:
    """Mixin providing the detection-hook helpers shared by all AI services.

    Subclasses override the crop constants and `EVENT_FOLDER`; services
    that never save images (e.g. face mask) inherit only `_find_parent`.
    """

    # Subfolder under /uploads for this service's event images. Required
    # only by services that call `_save_images_blocking`.
    EVENT_FOLDER = None

    #: Lớp model (kế thừa AiEventMixin) mà `save_event` chèn hàng vào.
    EVENT_MODEL = None
    #: CameraFilteredBroadcaster của loại này — nơi `save_event` bắn realtime.
    EVENT_BROADCASTER = None
    #: Nhãn ngắn cho log lỗi và cho `source` gửi sang engine lúc đánh thức ghi.
    EVENT_SOURCE = "ai"

    # Crop geometry. `CROP_PAD_*` is outward padding as a ratio of the
    # bbox width/height; `CROP_OUTPUT_*` is the fixed output size (aspect
    # preserved, letterboxed with `CROP_PAD_COLOR`). `CROP_VERTICAL_BIAS`
    # is "center" or "below" — see `fixed_size_crop`.
    CROP_PAD_LEFT = 0.2
    CROP_PAD_RIGHT = 0.2
    CROP_PAD_TOP = 0.2
    CROP_PAD_BOTTOM = 0.2
    CROP_OUTPUT_W = 400
    CROP_OUTPUT_H = 480
    CROP_PAD_COLOR = 114  # neutral grey, matches YOLO letterboxing
    CROP_VERTICAL_BIAS = "center"

    @staticmethod
    def _find_parent(meta, tid):
        """The detection in this frame carrying `tid`, or None.

        `process_ai_service` tags tracker ids back onto the raw detection
        dicts, so this is how a hook gets from an id to its bbox/score."""
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    @classmethod
    def _make_crop(cls, img, bx1, by1, bx2, by2):
        """Fixed-size crop around a bbox using this service's geometry."""
        return fixed_size_crop(
            img, bbox=(bx1, by1, bx2, by2),
            pad_lrtb=(cls.CROP_PAD_LEFT, cls.CROP_PAD_RIGHT,
                      cls.CROP_PAD_TOP, cls.CROP_PAD_BOTTOM),
            output_size=(cls.CROP_OUTPUT_W, cls.CROP_OUTPUT_H),
            pad_color=cls.CROP_PAD_COLOR,
            vertical_bias=cls.CROP_VERTICAL_BIAS,
        )

    @classmethod
    def _stem_suffix(cls, value):
        """Filename part after the frame seq. Tracker id by default;
        plate recognition overrides it with the sanitised plate text."""
        return str(int(value))

    @staticmethod
    def _normalized_box(parent, width, height):
        """bbox thô -> khung CHUẨN HOÁ [0,1] theo kích thước ảnh full, để
        frontend vẽ box lên ảnh ở bất kỳ kích thước hiển thị nào. Trả None nếu
        thiếu toạ độ hoặc ảnh suy biến."""
        if not width or not height:
            return None
        try:
            x1 = float(parent.get("x1", 0.0))
            y1 = float(parent.get("y1", 0.0))
            x2 = float(parent.get("x2", 0.0))
            y2 = float(parent.get("y2", 0.0))
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        def c(v):
            return max(0.0, min(1.0, v))

        return {
            "x1": c(x1 / width), "y1": c(y1 / height),
            "x2": c(x2 / width), "y2": c(y2 / height),
        }

    @classmethod
    def _save_images_blocking(cls, full_jpeg, meta, parent, stem_value):
        """Write the full frame and its crop; return (full_url, crop_url, box).

        `box` is the detection bbox normalised to [0,1] over the full frame
        (or None). Returns None when there's nothing renderable (no bytes,
        undecodable JPEG, degenerate crop, failed write) so callers can skip
        persisting an event with dangling image paths. Blocking on purpose —
        callers run it through `asyncio.to_thread`."""
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

        h, w = img.shape[:2]
        box = cls._normalized_box(parent, w, h)

        date = datetime.date.today().isoformat()
        folder_rel = os.path.join(cls.EVENT_FOLDER, str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        stem = f"{int(meta['seq']):010d}_{cls._stem_suffix(stem_value)}"
        full_abs = os.path.join(folder_abs, f"{stem}_full.jpg")
        crop_abs = os.path.join(folder_abs, f"{stem}_crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(full_jpeg)
        if not cv2.imwrite(crop_abs, crop):
            return None
        return (
            f"/uploads/{folder_rel}/{stem}_full.jpg",
            f"/uploads/{folder_rel}/{stem}_crop.jpg",
            box,
        )

    # ─── Lưu một sự kiện: ảnh → hàng DB → WebSocket → ghi hình ──────────
    #
    # Bốn service đã có bốn bản `_persist_event` chỉ khác nhau ở TÊN MODEL và
    # vài cột riêng. Gộp về đây để thêm một loại sự kiện không còn phải chép
    # lại cả khối try/except + to_thread + broadcast.

    async def save_event(
        self,
        meta,
        parent,
        full_jpeg,
        stem_value,
        timestamp,
        columns=None,
        payload=None,
        confidence=None,
    ):
        """Ghi một sự kiện của loại này và trả về hàng đã lưu (hoặc None).

        `columns` là các cột RIÊNG của loại (plate_number, mask_status…);
        `payload` là các trường riêng thêm vào gói WebSocket. `confidence` để
        ghi đè điểm tin cậy: khuôn mặt lưu ĐỘ GIỐNG với người đã đăng ký, không
        phải điểm phát hiện của box.

        Trả None khi không có gì để lưu (thiếu ảnh, crop suy biến, chưa có
        session factory) — chỗ gọi không cần phân biệt vì sự kiện thiếu ảnh
        thì lưu cũng chỉ ra một thẻ trống.
        """
        from app.services.process_ai_service import process_ai_service
        session_factory = process_ai_service._session_factory
        if session_factory is None:
            return None
        try:
            paths = await asyncio.to_thread(
                self._save_images_blocking, full_jpeg, meta, parent, stem_value,
            )
            if paths is None:
                return None
            full_url, crop_url, box = paths

            event = self.EVENT_MODEL(
                camera_id=str(meta["cameraId"]),
                confidence=float(
                    parent.get("score", 0.0) if confidence is None else confidence
                ),
                timestamp=int(timestamp),
                image_full=full_url,
                image_crop=crop_url,
                box_x1=box["x1"] if box else None,
                box_y1=box["y1"] if box else None,
                box_x2=box["x2"] if box else None,
                box_y2=box["y2"] if box else None,
                **(columns or {}),
            )
            async with session_factory() as db:
                db.add(event)
                await db.commit()

            # session_factory đặt expire_on_commit=False nên event.id vẫn còn
            # sau commit, không cần refresh thêm một vòng.
            if self.EVENT_BROADCASTER is not None:
                self.EVENT_BROADCASTER.publish({
                    "id": event.id,
                    "camera_id": event.camera_id,
                    "confidence": float(event.confidence),
                    "timestamp": int(event.timestamp),
                    "image_full": event.image_full,
                    "image_crop": event.image_crop,
                    "box": box,
                    **(payload or {}),
                })

            self.arm_recording(str(meta["cameraId"]))
            return event
        except Exception as exc:
            print(f"{self.EVENT_SOURCE} persist error: {exc}", file=sys.stderr)
            return None

    # ─── Ghi hình theo sự kiện AI ───────────────────────────────────────

    def arm_recording(self, camera_id: str) -> None:
        """Báo engine "camera này vừa có sự kiện AI" để nó GIỮ đoạn ghi.

        Gọi cho MỌI sự kiện, không có công tắc riêng từng AI. "Chỉ ghi khi có
        sự kiện" là một cài đặt của CAMERA (chế độ ghi 'motion' + ghi
        trước/ghi sau), nên chính engine mới là chỗ quyết định — nó vứt mọi
        đoạn không có gì xảy ra và lời gọi này giữ lại đoạn quanh sự kiện, đúng
        cơ chế mà chuyển động vẫn dùng.

        Camera đang "Luôn ghi" thì engine trả về "không có đoạn nào để giữ" và
        chẳng tốn gì thêm. Một sự kiện AI đã qua khử trùng theo track nên tần
        suất rất thấp — không đáng để thêm một công tắc nữa chỉ để tiết kiệm
        một lời gọi HTTP nội bộ.

        Fire-and-forget: sự kiện đã nằm trong DB rồi, engine không trả lời
        được thì cũng không được để mất sự kiện.
        """
        asyncio.create_task(self._arm_recording_call(camera_id))

    async def _arm_recording_call(self, camera_id: str) -> None:
        from app.api.httpx_client import HTTPXClient
        try:
            await HTTPXClient.post(
                f"/cameras/{camera_id}/ai-event",
                json={"source": self.EVENT_SOURCE},
            )
        except Exception as exc:
            print(f"{self.EVENT_SOURCE} arm recording failed: {exc}", file=sys.stderr)
