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



# Trạng thái xác nhận của từng tracker. entered_zone khởi tạo, in_the_area
# chạy vòng thử lại cho tới khi đọc được chuỗi biển đủ dài, exited_zone xoá.
# Khoá theo (camera_id, tracker_id).
_PENDING = "pending"     # chưa đọc đủ số ký tự tối thiểu; tiếp tục thử
_RESOLVED = "resolved"   # đã xác nhận biển và lưu sự kiện; ngừng thử

# Chuỗi biển sau khi lọc phải đạt số ký tự tối thiểu (chữ + số, bỏ khoảng
# trắng) thì mới đủ tin cậy để ghi một dòng vào DB. Đọc ngắn hơn thì giữ ở
# trạng thái PENDING — thường chỉ là tracker bắt được xe ở góc che mất một
# phần biển, frame sau sẽ đọc đủ.
#
# Không còn giá trị mặc định trong code: ngưỡng lấy TỪ extra_data
# ["min_plate_length"] của AI job (UI: "Số ký tự tối thiểu để lưu"), job nào
# chưa có thì KHÔNG ghi sự kiện nào cho tới khi được cấu hình. Thà ngừng ghi
# còn hơn tự ý chọn một con số rồi ghi ra dữ liệu mà không ai biết nó dựa
# trên ngưỡng nào.
#
# ĐỪNG nhầm với PlateWhiteListSettings.min_plate_length: cái đó là ngưỡng
# của nhánh mở barrier, thường dễ hơn vì đọc thiếu một ký tự vẫn đủ nhận ra
# xe đã đăng ký, còn ngưỡng ở đây quyết định có ghi EventPlate hay không.
# Nhánh whitelist / mở barrier có đủ ngưỡng riêng của nó (độ dài tối thiểu,
# độ tin cậy OCR, sai số ký tự, thời gian chờ) trong PlateWhiteListSettings
# theo từng camera, do plate_white_list_service tự áp dụng.

class PlateRecognitionService(AIServiceBase):
    # (camera_id, tracker_id) -> thời điểm lưu EventPlate gần nhất của tracker
    # đó. _track_state đã chặn trùng khi xe còn đứng trong vùng; biến này lo
    # trường hợp ra-vào lại (xe rời vùng một lúc hoặc mất detection) khiến
    # _track_state bị xoá nhưng tracker vẫn giữ nguyên id — nếu không sẽ ghi
    # thêm một dòng nữa. Để dài hơn cửa sổ của khuôn mặt vì xe đỗ có thể nằm
    # lì ở ranh giới vùng.
    _REENTER_COOLDOWN_S = 30

    def __init__(self):
        # (camera_id, tracker_id) -> _PENDING / _RESOLVED. Sống xuyên suốt các
        # frame của cùng một tracker; bị xoá khi exited_zone.
        self._track_state: dict = {}
        self._last_saved: dict = {}
        # camera_id đã in cảnh báo "AI job chưa có min_plate_length".
        self._warned_unconfigured = set()

    @staticmethod
    def _clean_plate(text_plate: str) -> str:
        # Bỏ dấu câu/ký hiệu nhưng GIỮ chữ tiếng Việt có dấu (một số mã tỉnh
        # có dấu) và khoảng trắng; độ dài sau đó được đo trên dạng đã chuẩn
        # hoá này.
        return re.sub(r'[^a-zA-Z0-9À-ỹ\s]', '', text_plate or "")

    async def plate_recognition(self, db: AsyncSession, req: PlateRecognitionDTO):
        # pre_time (và các ngưỡng whitelist khác) đã rời khỏi đây, sang bảng
        # plate_white_list_settings. Còn lại min_plate_length vì nó thuộc về
        # nhánh lưu EventPlate của chính AI job này.
        # Client không gửi thì không ghi khoá — job vẫn ở trạng thái chưa cấu
        # hình thay vì nhận một con số do backend tự chọn.
        extra_data = (
            {"min_plate_length": req.min_plate_length}
            if req.min_plate_length is not None
            else None
        )
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

    # `_find_parent` / `_make_crop` / `_save_images_blocking` lấy từ
    # AIServiceBase; chỉ khác nhau ở hình học crop, thư mục upload và hậu tố
    # tên file.
    EVENT_FOLDER = "plates"

    # Bbox biển số từ detection bám sát mép biển. Nới ra trước khi lưu để ảnh
    # crop thấy thêm một phần thân xe xung quanh — người xem lại dễ xác định
    # được xe nào. Nới hai bên nhiều hơn vì biển số có tỉ lệ rất ngang.
    CROP_PAD_LEFT = 0.4
    CROP_PAD_RIGHT = 0.4
    CROP_PAD_TOP = 0.3
    CROP_PAD_BOTTOM = 0.3

    # Mọi ảnh crop biển số đều xuất ra đúng kích thước này. Tỉ lệ 4:1 khớp
    # biển ô tô một hàng của Việt Nam (470×110 mm = 4.27:1) nên đa số ảnh lấp
    # đầy khung; biển xe máy hai hàng (~1.4:1) sẽ được mở rộng thêm nguồn theo
    # chiều ngang rồi căn giữa — không bao giờ bị kéo méo.
    CROP_OUTPUT_W = 280
    CROP_OUTPUT_H = 120

    @classmethod
    def _stem_suffix(cls, text_plate):
        # Ảnh biển số đặt tên theo chuỗi biển đọc được chứ không theo tracker
        # id, nên phải loại bỏ mọi ký tự không an toàn cho tên file.
        return re.sub(r"[^A-Za-z0-9_-]", "", text_plate) or "unknown"

    async def _persist_event(self, meta, parent, full_jpeg, text_plate, timestamp):
        # Import trễ để tránh phụ thuộc vòng lúc nạp module; tới lúc task này
        # chạy thì process_ai_service._session_factory đã được tạo trên đúng
        # event loop mà _persist_event đang chạy (loop của recv-loop).
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
            # session_factory đặt expire_on_commit=False nên event.id vẫn còn
            # giá trị sau khi commit mà không cần refresh. Đẩy đúng dòng đó
            # tới mọi client đang nghe WebSocket.
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

    @staticmethod
    def _min_plate_len(extra_data):
        """Ngưỡng số ký tự của AI job, hoặc None khi job chưa cấu hình.

        None cũng là kết quả cho giá trị hỏng ("abc", null, 0) — một
        extra_data lỗi không được phép biến mọi chuỗi OCR thành hợp lệ.
        """
        try:
            value = int((extra_data or {}).get("min_plate_length"))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _warn_unconfigured(self, camera_id: str) -> None:
        # Ngừng ghi sự kiện trong im lặng thì rất khó hiểu; in mỗi frame thì
        # ngập log. In đúng một lần cho mỗi camera.
        if camera_id in self._warned_unconfigured:
            return
        self._warned_unconfigured.add(camera_id)
        print(
            f"[plate] camera {camera_id} chua co min_plate_length trong AI job "
            f"-> khong ghi EventPlate. Vao Cau hinh AI -> Bien so, dat "
            f"'So ky tu toi thieu de luu' roi Luu lai."
        )

    def _try_confirm_plate(self, meta, parent, full_jpeg, timestamp, secondary_conf,
                           key, log_label, persist: bool = True, extra_data=None):
        """Đọc chuỗi biển số từ các OCR children của detection.

        Trả về True khi biển đã được xác nhận (đủ số ký tự tối thiểu cấu hình
        cho camera này), False khi vẫn cần thêm frame nữa. Đồng thời kích hoạt tác vụ
        whitelist/mở barrier cho MỌI lần đọc — kể cả đọc thiếu — vì ngưỡng
        của nhánh whitelist do PlateWhiteListSettings quyết định, độc lập với
        việc lưu DB.

        Khi `persist` = False thì vẫn đọc OCR và vẫn chạy whitelist/barrier
        (để cổng hoạt động ở mọi frame) nhưng KHÔNG ghi thêm dòng EventPlate
        và không đụng vào trạng thái — dùng khi tracker đã ở RESOLVED, nhằm
        tránh lưu trùng."""
        children = parent.get("children", [])
        # Nhánh whitelist/barrier chạy TRƯỚC và độc lập: nó tự đọc lại biển
        # bằng ocr_confidence riêng của camera, nên vẫn có thể mở cổng ở
        # những frame mà secondary_conf ở đây đọc ra chuỗi rỗng.
        if children:
            asyncio.create_task(
                plate_white_list_service.process_ai_result(
                    children, str(meta["cameraId"]),
                )
            )

        text_plate = detect_plate_from_children(children, secondary_conf)
        if not text_plate:
            return False
        t = self._clean_plate(text_plate)
        min_len = self._min_plate_len(extra_data)
        if min_len is None:
            self._warn_unconfigured(str(meta["cameraId"]))
            return False
        if len(t) < min_len:
            return False

        # Gửi thẳng children thay vì chuỗi biển đã dựng: bãi xe đọc lại bằng
        # ocr_confidence riêng của nó (xem ParkingLot.ocr_confidence).
        task_parking_lot.add_task({
                "task": "plate_recognition",
                "children": children,
                "timestamp": timestamp,
                "camera_id": meta["cameraId"],
                "full_jpeg": full_jpeg,
            })
        # Đã xác nhận và lưu ở frame trước: vẫn tiếp tục đọc và bắn whitelist
        # ở trên, nhưng không ghi thêm một EventPlate trùng.
        if not persist:
            return True
        # Chống trùng khi ra-vào lại: cùng một tracker có thể rời vùng trong
        # chốc lát (hoặc mất detection) làm _track_state bị xoá, rồi vào lại
        # với đúng id cũ. Bỏ qua dòng trùng nhưng vẫn đánh dấu RESOLVED để
        # in_the_area ngừng thử lại.
        last = self._last_saved.get(key)
        if last is not None and timestamp - last < self._REENTER_COOLDOWN_S:
            self._track_state[key] = _RESOLVED
            return True
        # Đã xác nhận — đổi trạng thái ĐỒNG BỘ trước khi chạy tác vụ lưu bất
        # đồng bộ, để in_the_area của frame ngay sau đó thấy _RESOLVED và
        # không lên lịch lưu trùng.
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
        # Khởi tạo trạng thái _PENDING để in_the_area tiếp quản nếu OCR của
        # frame này đọc chưa đủ. _try_confirm_plate sẽ đổi sang _RESOLVED khi
        # thành công.
        self._track_state[key] = _PENDING
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "entered_zone", extra_data=extra_data,
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        self._track_state.pop((str(meta["cameraId"]), int(id)), None)
        # Giữ lại các mốc thời gian lưu gần đây phục vụ cooldown ra-vào lại;
        # bỏ những mốc đã quá hạn để dict không phình vô hạn.
        cutoff = timestamp - self._REENTER_COOLDOWN_S
        self._last_saved = {k: t for k, t in self._last_saved.items() if t >= cutoff}
        print(f"Plate exited_zone")

    def in_the_area(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        key = (str(meta["cameraId"]), int(id))
        state = self._track_state.get(key)
        # Bỏ qua những tracker chưa từng vào vùng.
        if state is None:
            return
        parent = self._find_parent(meta, id)
        if parent is None:
            return
        # _PENDING -> vẫn đang xác nhận, thành công thì lưu. _RESOLVED -> vẫn
        # đọc/chạy whitelist mỗi frame nhưng không lưu thêm dòng trùng.
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "in_the_area", persist=(state == _PENDING),
            extra_data=extra_data,
        )



plate_recognition_service = PlateRecognitionService()
