import asyncio
import os
import sys
from dataclasses import replace as dc_replace
from typing import Optional

import requests

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ai_config import AIConfig

from app.dto.restricted_area_dto import RestrictedAreaDTO
from app.enum.config_ai_enum import TypeConfigAiEnum
from app.models.restricted_areas import RestrictedArea
from app.repositories.restricted_area_repository import RestrictedAreaRepository
from app.services.ai_job_service import AIJobSpec, AIStage, AIVariant, ai_job_service
from app.services.ai_service_base import AIServiceBase
from app.ws.restricted_area_event_ws import restricted_area_event_broadcaster

# Single-stage detection job (no stage-2 / no alignment). class_filter
# "0,1" tells the C++ engine to drop every YOLO class except 0 and 1
# before tracking — keeps unrelated objects out of the restricted-area
# events entirely, no Python-side filter needed.
RESTRICTED_AREA_SPEC = AIJobSpec(
    config_type=TypeConfigAiEnum.RESTRICTED_AREA.value,
    name="restricted area",
    stages=(
        AIStage(model_file="yolov8.rknn", model_type="yolov8_detect",
                class_filter="0"),
    ),
)

# RESTRICTED_AREA_SPEC = AIJobSpec(
#     config_type=TypeConfigAiEnum.RESTRICTED_AREA.value,
#     name="restricted area",
#     stages=(AIStage(model_file="yolov8-mask-s.rknn",
#                     model_type="yolov8_detect", class_filter="0,3,5"),),
# )

# RESTRICTED_AREA_SPEC = AIJobSpec(
#     config_type=TypeConfigAiEnum.RESTRICTED_AREA.value,
#     name="restricted area",
#     stages=(AIStage(model_file="rf_detr_m.rknn",
#                     model_type="rf_detect", class_filter="1"),),
# )

# ─── Gọi điện thoại báo động (voip24h) ───────────────────────────────
# Mỗi lần có người vào vùng cấm thì gọi API tổng hợp giọng nói của
# voip24h để nó gọi tới số trực. Mọi tham số đọc từ biến môi trường để
# đổi số/nội dung mà không phải sửa mã; giá trị mặc định là cấu hình
# đang dùng.
VOICE_ALERT_ENABLED = os.getenv("VOICE_ALERT_ENABLED", "1").lower() not in (
    "0", "false", "no", "off",
)
VOICE_ALERT_URL = os.getenv(
    "VOICE_ALERT_URL", "http://otp.voip24h.vn/api/voice_synthesis"
)
VOICE_ALERT_VOIP = os.getenv(
    "VOICE_ALERT_VOIP", "71d9486620f9d9b025afcc29fef48c0edf26ad84"
)
VOICE_ALERT_PHONE = os.getenv("VOICE_ALERT_PHONE", "0355536733")
VOICE_ALERT_TEXT = os.getenv("VOICE_ALERT_TEXT", "Có người vào khu vực cấm")
VOICE_ALERT_LANGUAGE = os.getenv("VOICE_ALERT_LANGUAGE", "vi-Vn")
# Cookie phiên trong ví dụ gốc. API xác thực bằng tham số `voip` nên
# cookie này thường không cần; giữ lại vì đó là request đã chạy được.
VOICE_ALERT_COOKIE = os.getenv("VOICE_ALERT_COOKIE", "PHPSESSID=66m5fvbq269hossuc4moscna71")
# Quá hạn thì bỏ cuộc — đây là mạng ngoài, treo lâu sẽ giữ thread vô ích.
VOICE_ALERT_TIMEOUT_S = float(os.getenv("VOICE_ALERT_TIMEOUT_S", "10"))


# Một cách làm; model cụ thể thì chọn được riêng trên giao diện (build_spec),
# nhưng cách LÀM thì chỉ có một: mọi hộp giữ lại đều được track, không có lớp
# phụ nào gắn vào. Nên giao diện không hiện ô chọn loại.
RESTRICTED_AREA_VARIANT = AIVariant(
    id="detect_track_all",
    label="Phát hiện + bám mọi lớp giữ lại",
    spec=RESTRICTED_AREA_SPEC,
)


class RestrictedAreaService(AIServiceBase):
    VARIANTS = (RESTRICTED_AREA_VARIANT,)

    # Lọc lớp cho tracker (track_classes của biến thể) khác với
    # AIStage.class_filter: class_filter cắt ngay trong engine C++ nên lớp bị
    # bỏ không có mặt trong meta luôn, còn track_classes chỉ chặn ở ĐẦU VÀO
    # tracker — meta vẫn liệt kê đủ mọi lớp model thấy và overlay gỡ lỗi vẫn
    # vẽ chúng. Vùng cấm dùng class_filter (lọc từ engine) nên track hết.

    # (camera_id, tracker_id) -> timestamp of the last event written for that
    # tracker. A tracker that exits and re-enters within this window (brief
    # occlusion, boundary loiter) is treated as the same presence and does not
    # produce a second row. This is the only dedup for restricted-area events.
    _REENTER_COOLDOWN_S = 15

    # Khoảng nghỉ giữa hai CUỘC GỌI của cùng một camera. Tách khỏi
    # _REENTER_COOLDOWN_S (vốn tính theo từng tracker): mười người cùng bước
    # vào là mười lần entered_zone, không chặn ở đây thì thành mười cuộc gọi
    # liên tiếp tới cùng một số. Sự kiện/ảnh vẫn được lưu đủ, chỉ cuộc gọi mới
    # bị gộp.
    _VOICE_CALL_COOLDOWN_S = float(os.getenv("VOICE_ALERT_COOLDOWN_S", "60"))

    def __init__(self):
        self._last_saved: dict = {}
        # camera_id -> timestamp của cuộc gọi gần nhất
        self._last_voice_call: dict = {}

    @staticmethod
    def _infer_model_type(file_name: str) -> str:
        """Đoán model_type từ tên file khi giao diện không gửi kèm.
        rf_detr/rf_det* dùng đầu ra khác YOLO nên engine cần "rf_detect"."""
        name = (file_name or "").lower()
        if name.startswith("rf_") or "rf_detr" in name or "rf_det" in name:
            return "rf_detect"
        return "yolov8_detect"

    # Vùng cấm chỉ có MỘT tầng, nên "tầng gốc" ở đây luôn là stages[0].
    @staticmethod
    def _root() -> AIStage:
        return RESTRICTED_AREA_SPEC.stages[0]

    @classmethod
    def build_spec(cls, req: RestrictedAreaDTO) -> AIJobSpec:
        """Spec ĐỘNG theo lựa chọn trên giao diện; trường nào bỏ trống thì lấy
        mặc định của RESTRICTED_AREA_SPEC (AIJobSpec là frozen dataclass)."""
        root = cls._root()
        model_file = (req.modelFile or "").strip() or root.model_file
        model_type = (req.modelType or "").strip() or (
            root.model_type
            if model_file == root.model_file
            else cls._infer_model_type(model_file)
        )
        # classFilter: None = không gửi -> giữ mặc định; "" = giữ TẤT CẢ lớp.
        class_filter = (
            root.class_filter
            if req.classFilter is None
            else req.classFilter.strip()
        )
        return dc_replace(
            RESTRICTED_AREA_SPEC,
            stages=(dc_replace(root, model_file=model_file,
                               model_type=model_type,
                               class_filter=class_filter),),
        )

    async def restricted_area(self, db: AsyncSession, req: RestrictedAreaDTO):
        spec = self.build_spec(req)
        root = spec.stages[0]
        # Lưu lựa chọn vào ai_configs.extra_data để giao diện nạp lại đúng
        # model/lớp đang chạy (engine chỉ giữ path, không giữ tên file).
        extra = {
            "modelFile": root.model_file,
            "modelType": root.model_type,
            "classFilter": root.class_filter or "",
        }
        return await ai_job_service.upsert(db, req, spec, extra_data=extra)

    async def get_settings(self, db: AsyncSession, camera_id: str) -> dict:
        """Model/lớp đang áp cho camera này + mặc định (cho giao diện hiển thị)."""
        row = (await db.execute(
            select(AIConfig).where(
                AIConfig.camera_id == camera_id,
                AIConfig.type == TypeConfigAiEnum.RESTRICTED_AREA.value,
            )
        )).scalars().first()
        extra = (row.extra_data if row and isinstance(row.extra_data, dict) else {}) or {}
        root = self._root()
        return {
            "modelFile": extra.get("modelFile") or root.model_file,
            "modelType": extra.get("modelType") or root.model_type,
            "classFilter": extra.get("classFilter")
            if extra.get("classFilter") is not None
            else (root.class_filter or ""),
            "defaults": {
                "modelFile": root.model_file,
                "modelType": root.model_type,
                "classFilter": root.class_filter or "",
            },
        }

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
    # `_find_parent` / `_make_crop` / `_save_images_blocking` come from
    # AIServiceBase; only the crop geometry and upload folder differ.
    EVENT_FOLDER = "restricted"

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

    # Lưu ảnh + hàng + WebSocket + đánh thức ghi hình: toàn bộ nằm ở
    # AIServiceBase.save_event. Ở đây chỉ còn cái RIÊNG của vùng cấm —
    # class_id đi kèm gói realtime (chưa có cột trong bảng) để giao diện đặt
    # nhãn khác nhau cho từng lớp YOLO.
    EVENT_MODEL = RestrictedArea
    EVENT_BROADCASTER = restricted_area_event_broadcaster
    EVENT_SOURCE = "restricted_area"

    async def _persist_event(self, meta, parent, full_jpeg, tid, timestamp,
                             extra_data=None):
        await self.save_event(
            meta, parent, full_jpeg, tid, timestamp,
            payload={"class_id": parent.get("classId")},
            extra_data=extra_data,
        )

    @staticmethod
    def _voice_call_blocking(camera_id: str) -> None:
        """Gọi API voip24h. CHẠY TRONG THREAD RIÊNG — requests là blocking, gọi
        thẳng trên vòng lặp sự kiện sẽ đứng toàn bộ đường ống AI trong lúc chờ
        mạng."""
        payload = {
            "voip": VOICE_ALERT_VOIP,
            "phone": VOICE_ALERT_PHONE,
            "data_speech": VOICE_ALERT_TEXT,
            "LanguageCode": VOICE_ALERT_LANGUAGE,
        }
        headers = {"Cookie": VOICE_ALERT_COOKIE} if VOICE_ALERT_COOKIE else {}
        try:
            response = requests.post(
                VOICE_ALERT_URL,
                headers=headers,
                data=payload,
                timeout=VOICE_ALERT_TIMEOUT_S,
            )
            print(
                f"restricted_area voice call camera={camera_id} "
                f"phone={VOICE_ALERT_PHONE} status={response.status_code} "
                f"body={response.text[:300]}"
            )
        except Exception as exc:
            # Không ném ra ngoài: cuộc gọi hỏng thì vẫn phải ghi sự kiện.
            print(f"restricted-area voice call error: {exc}", file=sys.stderr)

    def _fire_voice_call(self, camera_id: str, timestamp: float) -> None:
        if not VOICE_ALERT_ENABLED:
            return
        last = self._last_voice_call.get(camera_id)
        if last is not None and timestamp - last < self._VOICE_CALL_COOLDOWN_S:
            return
        # Đóng dấu TRƯỚC khi gọi, để hai sự kiện sát nhau trong cùng một frame
        # không cùng lọt qua.
        self._last_voice_call[camera_id] = timestamp
        asyncio.create_task(asyncio.to_thread(self._voice_call_blocking, camera_id))

    def entered_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        key = (str(meta["cameraId"]), int(id))
        last = self._last_saved.get(key)
        if last is not None and timestamp - last < self._REENTER_COOLDOWN_S:
            return
        # Stamp synchronously (before the async write) so a re-enter on the
        # very next frame is suppressed even while the row is still being saved.
        self._last_saved[key] = timestamp
        print(f"restricted_area entered_zone id={id} class={parent.get('classId')}")
        asyncio.create_task(
            self._persist_event(meta, parent, full_jpeg, id, timestamp, extra_data)
        )
        # self._fire_voice_call(str(meta["cameraId"]), timestamp)

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"restricted_area dwell_alert id={id}")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        # Keep recent stamps so a quick re-enter is still deduped, but drop
        # aged-out ones so the map can't grow without bound.
        cutoff = timestamp - self._REENTER_COOLDOWN_S
        self._last_saved = {k: t for k, t in self._last_saved.items() if t >= cutoff}
        print(f"restricted_area exited_zone id={id}")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        # No re-persist while the tracker stays — entered_zone already
        # captured a row. dwell_alert handles "stayed too long".
        # print(meta)
        pass


restricted_area_service = RestrictedAreaService()
