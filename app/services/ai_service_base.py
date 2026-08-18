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
import base64
import datetime
import os
import sys

import cv2
import numpy as np

from app.services.ai_job_service import AIJobService
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

    # ─── Biến thể (cách làm) của loại AI này ──────────────────────────
    #
    # Tuple các AIVariant, biến thể ĐẦU TIÊN là mặc định. Loại AI nào cũng
    # phải khai ít nhất một biến thể — kể cả khi chỉ có đúng một cách làm —
    # để mọi thứ phía sau (tracking, gắn lớp phụ, overlay) đọc từ CÙNG một
    # chỗ thay vì rải rác thành thuộc tính lớp.
    VARIANTS: tuple = ()

    @classmethod
    def variant(cls, extra_data=None):
        """Biến thể đang áp cho một camera, theo extra_data["variant"].

        Không khai / khai id lạ (biến thể bị xoá khỏi code sau khi camera đã
        lưu) đều rơi về biến thể đầu tiên: thà chạy cách mặc định còn hơn tắt
        AI của camera đó."""
        if not cls.VARIANTS:
            return None
        wanted = (extra_data or {}).get("variant")
        for v in cls.VARIANTS:
            if v.id == wanted:
                return v
        return cls.VARIANTS[0]

    @classmethod
    def resolve_variant(cls, req):
        """Biến thể theo lựa chọn gửi lên từ giao diện (`req.variant`).

        Không gửi = dùng mặc định, nên client cũ và loại AI chỉ có một cách làm
        đều chạy y như trước mà không phải sửa gì."""
        return cls.variant({"variant": getattr(req, "variant", None)})

    @classmethod
    def variant_options(cls):
        """Danh sách biến thể cho giao diện. Một phần tử thì giao diện không
        hiện ô chọn — không có gì để chọn.

        Kèm luôn CÂY MODEL của từng biến thể: trang thử model nạp thẳng cây này
        vào form để chạy đúng thứ camera đang chạy, thay vì bắt người dùng gõ
        lại tay rồi tự hỏi vì sao kết quả khác. Ô chọn cách xử lý chỉ đọc
        id/label nên thêm khoá này không ảnh hưởng gì tới nó."""
        return [
            {
                "id": v.id,
                "label": v.label,
                "stages": [
                    AIJobService.stage_preview(s, i)
                    for i, s in enumerate(v.spec.stages)
                ],
            }
            for v in cls.VARIANTS
        ]

    @staticmethod
    def containment(outer, inner):
        """Phần diện tích của `inner` nằm trong `outer` (0..1).

        KHÔNG phải IoU: box phụ (khẩu trang, biển số) luôn nhỏ hơn hẳn box
        chính (người, xe) nên IoU luôn bé và không phân biệt được gì. Cái cần
        biết là "cái nhỏ có nằm gọn trong cái lớn không"."""
        ix1 = max(outer.get("x1", 0), inner.get("x1", 0))
        iy1 = max(outer.get("y1", 0), inner.get("y1", 0))
        ix2 = min(outer.get("x2", 0), inner.get("x2", 0))
        iy2 = min(outer.get("y2", 0), inner.get("y2", 0))
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inner_area = ((inner.get("x2", 0) - inner.get("x1", 0)) *
                      (inner.get("y2", 0) - inner.get("y1", 0)))
        if inner_area <= 0:
            return 0.0
        return ((ix2 - ix1) * (iy2 - iy1)) / inner_area

    @classmethod
    def find_attached(cls, meta, parent, extra_data=None):
        """Box phụ thuộc về box được track này, hoặc None.

        Chấm điểm bằng độ tin cậy × tỉ lệ nằm trong, nên giữa hai biển số
        cùng lọt vào khung một chiếc xe thì cái nằm gọn hơn và chắc hơn thắng.

        MỘT VẬT PHỤ CHỈ THUỘC VỀ MỘT VẬT CHÍNH. Đây không phải chuyện sạch sẽ
        lý thuyết: model xe hay vẽ hai hộp chồng nhau lên cùng một chiếc xe
        (đo trên 22 biển thật thì 14% nằm gọn trong HAI hộp xe, lớp [1,4],
        [2,2], [3,1]). Không có luật sở hữu thì cả hai xe đều nhận cùng một
        biển, mỗi xe là một track, và một lượt xe vào đẻ ra hai sự kiện chữ
        giống hệt nhau. Nên sau khi chọn được box phụ hợp nhất, còn phải kiểm
        lại chính mình có phải chủ của nó không; không phải thì thử box phụ
        kế tiếp, hết thì thôi.

        Biến thể không khai attach_classes trả về None — lúc đó chính box được
        track đã là vật cần xử lý rồi (biển số tự nó là một track)."""
        variant = cls.variant(extra_data)
        if not variant or not variant.attach_classes or parent is None:
            return None

        candidates = []
        for det in meta.get("detections", []):
            if det.get("classId") not in variant.attach_classes:
                continue
            ratio = cls.containment(parent, det)
            if ratio < variant.attach_containment:
                continue
            candidates.append((float(det.get("score", 0.0)) * ratio, det))

        candidates.sort(key=lambda item: -item[0])
        for _, det in candidates:
            if cls._attach_owner(meta, det, variant) is parent:
                return det
        return None

    @classmethod
    def _attach_owner(cls, meta, attached, variant):
        """Vật chính SỞ HỮU một vật phụ: hộp chứa nó nhiều nhất.

        Hoà nhau (biển nằm gọn trong cả hộp xe máy lẫn hộp xe tải chồng lên
        nó) thì hộp NHỎ HƠN thắng — biển gắn trên cái xe ôm sát nó, không phải
        cái hộp to bao ngoài."""
        best, best_key = None, None
        for det in meta.get("detections", []):
            class_id = det.get("classId")
            if class_id in variant.attach_classes:
                continue
            if (variant.track_classes is not None
                    and class_id not in variant.track_classes):
                continue
            ratio = cls.containment(det, attached)
            if ratio < variant.attach_containment:
                continue
            area = ((det.get("x2", 0) - det.get("x1", 0)) *
                    (det.get("y2", 0) - det.get("y1", 0)))
            key = (ratio, -area)
            if best_key is None or key > best_key:
                best, best_key = det, key
        return best

    @classmethod
    def subject(cls, meta, parent, extra_data=None):
        """Vật thật sự cần xử lý cho một tracker id, hoặc None.

        Biến thể có gắn lớp phụ (track xe, biển gắn vào xe) thì đó là box phụ,
        và KHÔNG rơi về box được track: một chiếc xe không mang biển nào thì
        chẳng có gì để đọc, trả về xe chỉ khiến chỗ gọi đi đọc nhầm. Biến thể
        không gắn gì thì chính box được track là vật cần xử lý."""
        variant = cls.variant(extra_data)
        if variant and variant.attach_classes:
            return cls.find_attached(meta, parent, extra_data)
        return parent

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

    @staticmethod
    def should_save_events(extra_data) -> bool:
        """Cấu hình (camera, loại AI) này có GHI sự kiện xuống DB/đĩa không.

        Thiếu khoá = BẬT. Cấu hình lưu trước khi có cột, và mọi đường gọi chưa
        kịp truyền extra_data xuống, đều phải giữ nguyên hành vi cũ — mất sự
        kiện âm thầm tệ hơn nhiều so với ghi thừa."""
        return bool((extra_data or {}).get("save_events", True))

    @classmethod
    def _render_crop_blocking(cls, full_jpeg, parent):
        """Ảnh crop dạng data-URL + box, KHÔNG chạm đĩa.

        Dùng khi camera tắt ghi sự kiện: thẻ trên bảng sự kiện trực tiếp vẫn
        cần một tấm ảnh để nhìn, mà cả điểm của việc tắt là không để lại gì.
        Chỉ gửi CROP (vài KB) chứ không gửi khung hình đầy đủ — sự kiện đã khử
        trùng theo track nên thưa, nhưng nhét vài trăm KB base64 vào mỗi gói
        WebSocket thì vẫn là phí vô ích, và modal xem chi tiết vốn đã rơi về
        ảnh crop khi không có ảnh toàn cảnh."""
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
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        h, w = img.shape[:2]
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        return data_url, cls._normalized_box(parent, w, h)

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
        extra_data=None,
    ):
        """Ghi một sự kiện của loại này và trả về hàng đã lưu (hoặc None).

        `columns` là các cột RIÊNG của loại (plate_number, mask_status…);
        `payload` là các trường riêng thêm vào gói WebSocket. `confidence` để
        ghi đè điểm tin cậy: khuôn mặt lưu ĐỘ GIỐNG với người đã đăng ký, không
        phải điểm phát hiện của box.

        `extra_data` mang cấu hình của cặp (camera, loại AI). Khi nó TẮT việc
        ghi sự kiện thì hàm này không đụng vào DB lẫn đĩa nhưng VẪN bắn gói
        realtime (kèm ảnh crop dạng data-URL) — camera đó vẫn xem được sự kiện
        đang xảy ra, chỉ là không tra cứu lại được về sau. Lúc đó trả về None
        vì thật sự không có hàng nào.

        Trả None khi không có gì để lưu (thiếu ảnh, crop suy biến, chưa có
        session factory) — chỗ gọi không cần phân biệt vì sự kiện thiếu ảnh
        thì lưu cũng chỉ ra một thẻ trống.
        """
        from app.services.process_ai_service import process_ai_service
        session_factory = process_ai_service._session_factory
        if session_factory is None:
            return None
        try:
            if not self.should_save_events(extra_data):
                return await self._broadcast_transient(
                    meta, parent, full_jpeg, timestamp, columns, payload, confidence,
                )

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

    async def _broadcast_transient(self, meta, parent, full_jpeg, timestamp,
                                   columns, payload, confidence):
        """Sự kiện CHỈ realtime: bắn đi rồi thôi, không DB không đĩa.

        Gói giữ nguyên hình dạng của sự kiện đã lưu để bảng sự kiện không phải
        biết hai loại, chỉ khác `id = None` (chưa từng có hàng nào) và
        `transient = True` cho phía giao diện muốn đánh dấu "không lưu lại".
        Các cột riêng của loại (plate_number, mask_status…) vẫn đi kèm, vì đó
        chính là nội dung người dùng cần đọc trên thẻ.
        """
        if self.EVENT_BROADCASTER is None:
            return None
        rendered = await asyncio.to_thread(
            self._render_crop_blocking, full_jpeg, parent,
        )
        crop_url, box = rendered if rendered else ("", None)
        self.EVENT_BROADCASTER.publish({
            "id": None,
            "camera_id": str(meta["cameraId"]),
            "confidence": float(
                parent.get("score", 0.0) if confidence is None else confidence
            ),
            "timestamp": int(timestamp),
            "image_full": "",
            "image_crop": crop_url,
            "box": box,
            "transient": True,
            **(columns or {}),
            **(payload or {}),
        })
        # KHÔNG gọi arm_recording: giữ đoạn video là chuyện của chế độ ghi bên
        # camera, mà người dùng vừa nói rõ camera này không cần lưu lại gì của
        # AI. Giữ lại đoạn ghi vì một sự kiện cố tình không lưu là làm ngược ý.
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
