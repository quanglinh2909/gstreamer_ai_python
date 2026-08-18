"""Phần dùng chung của mọi cách đọc biển số.

Xem `__init__.py` của gói này để biết bố cục: lớp cơ sở ở đây, mỗi cách đọc
biển một file riêng, và mặt tiền + bảng tra biến thể ở `__init__.py`.
"""

import asyncio
import re
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.dto.plate_recognition_dto import PlateRecognitionDTO
from app.models.event_plate import EventPlate
from app.repositories.event_plate_repository import EventPlateRepository
from app.services.ai_job_service import ai_job_service
from app.services.ai_service_base import AIServiceBase
from app.services.plate_white_list_service import plate_white_list_service
from app.tasks.task_parking_lot import task_parking_lot
from app.utils.plate_recognition_hepper import (
    flatten_char_boxes,
    looks_like_vn_plate,
    plate_text_from_detection,
)
from app.ws.plate_event_ws import plate_event_broadcaster

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


class PlateRecognitionBase(AIServiceBase):
    """Phần dùng chung của MỌI cách đọc biển số.

    Vòng đời một lượt xe (vào vùng → thử đọc lại từng khung → xác nhận → lưu →
    rời vùng), chống trùng, whitelist/barrier, hình học ảnh crop, bảng sự kiện:
    tất cả giống hệt nhau ở cả hai cách làm, nên nằm hết ở đây.

    Lớp con chỉ khai BIẾN THỂ của nó. Mọi khác biệt còn lại đã được diễn tả
    bằng dữ liệu trong AIVariant (cây model, track lớp nào, biển gắn vào đâu)
    nên lớp con không phải viết lại một dòng logic nào — chỗ nào cần biết
    "vật cần đọc là gì" thì gọi self.subject(...)."""

    # (camera_id, tracker_id) -> thời điểm lưu EventPlate gần nhất của tracker
    # đó. _track_state đã chặn trùng khi xe còn đứng trong vùng; biến này lo
    # trường hợp ra-vào lại (xe rời vùng một lúc hoặc mất detection) khiến
    # _track_state bị xoá nhưng tracker vẫn giữ nguyên id — nếu không sẽ ghi
    # thêm một dòng nữa. Để dài hơn cửa sổ của khuôn mặt vì xe đỗ có thể nằm
    # lì ở ranh giới vùng.
    _REENTER_COOLDOWN_S = 30

    # Số khung phải đọc RA CÙNG một chuỗi thì mới ghi ngay. Đo trên 110 lượt xe
    # thật (log `[plate-doc]`): 45% số lượt có ít nhất hai khung trùng nhau, nên
    # ngưỡng 2 bắt được gần một nửa số lượt ngay tại chỗ, phần còn lại rơi xuống
    # đường chốt-lúc-rời-vùng ở dưới.
    _VOTES_TO_CONFIRM = 2

    def __init__(self):
        # (camera_id, tracker_id) -> _PENDING / _RESOLVED. Sống xuyên suốt các
        # frame của cùng một tracker; bị xoá khi exited_zone.
        self._track_state: dict = {}
        # (camera_id, tracker_id) -> {chuoi_bien: so_phieu}. Xem _cast_vote.
        self._votes: dict = {}
        # (camera_id, tracker_id) -> (chuoi, subject, full_jpeg, timestamp) của
        # ứng viên ĐANG DẪN ĐẦU, để còn ghi được khi xe rời vùng mà chưa đủ phiếu.
        self._leader: dict = {}
        self._last_saved: dict = {}
        # (camera_id, chuoi_bien) -> lan luu gan nhat. Chan trung khi cung mot
        # xe mang hai tracker id khac nhau; xem cho dung trong _try_confirm_plate.
        self._last_saved_plate: dict = {}
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
        variant = self.resolve_variant(req)
        # Biến thể phải nằm trong extra_data: nó quyết định track lớp nào và
        # đọc biển ở đâu, mà vòng nhận kết quả chỉ có mỗi ai_configs để tra.
        extra_data = {"variant": variant.id}
        if req.min_plate_length is not None:
            extra_data["min_plate_length"] = req.min_plate_length
        return await ai_job_service.upsert(
            db, req, variant.spec, extra_data=extra_data,
        )

    async def test_inference(
        self,
        image: tuple,
        primary_conf: float = 0.3,
        secondary_conf: float = 0.3,
        variant: Optional[str] = None,
    ):
        return await ai_job_service.inference_with_spec(
            image=image,
            spec=self.variant({"variant": variant}).spec,
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

    # Ghi ảnh + hàng + WebSocket + đánh thức ghi hình nằm ở
    # AIServiceBase.save_event; ở đây chỉ còn cột và trường RIÊNG của biển số.
    EVENT_MODEL = EventPlate
    EVENT_BROADCASTER = plate_event_broadcaster
    EVENT_SOURCE = "plate_recognition"

    async def _persist_event(self, meta, parent, full_jpeg, text_plate, timestamp,
                             extra_data=None):
        await self.save_event(
            meta, parent, full_jpeg, text_plate, timestamp,
            columns={"plate_number": text_plate},
            payload={
                "plate_number": text_plate,
                "whitelisted": plate_white_list_service.is_whitelisted(text_plate),
            },
            extra_data=extra_data,
        )

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
        # Biến thể bám theo XE thì vật cần đọc là cái BIỂN gắn vào xe đó, chứ
        # không phải hộp được track. Biến thể bám theo biển thì subject chính
        # là parent — nên đoạn dưới không phải phân biệt hai trường hợp.
        subject = self.subject(meta, parent, extra_data)
        if subject is None:
            return False
        # Ký tự có thể nằm ngay dưới biển (OCR một tầng) hoặc dưới từng dòng
        # chữ (PP-OCR hai tầng); gom phẳng về một danh sách ở toạ độ khung gốc.
        children = flatten_char_boxes(subject)
        # Nhánh whitelist/barrier chạy TRƯỚC và độc lập: nó tự đọc lại biển
        # bằng ocr_confidence riêng của camera, nên vẫn có thể mở cổng ở
        # những frame mà secondary_conf ở đây đọc ra chuỗi rỗng.
        if children:
            asyncio.create_task(
                plate_white_list_service.process_ai_result(
                    children, str(meta["cameraId"]),
                )
            )

        # secondary_conf là ngưỡng của TỪNG KÝ TỰ, không phải của cả biển: ký
        # tự nào dưới ngưỡng bị bỏ khỏi chuỗi. Model OCR hiện tại cho điểm ký
        # tự khoảng 0,66-0,85 nên đặt ngưỡng ~0,7 là cắt mất vài ký tự và chuỗi
        # còn lại vừa thiếu vừa sai thứ tự (mất ký tự làm hỏng luôn phép tách
        # hai dòng). Đo trên biển thật: ngưỡng 0,66 -> '53-79622' (đúng),
        # ngưỡng 0,70 -> '79562'. Để quanh 0,3.
        text_plate = plate_text_from_detection(subject, secondary_conf)
        if not text_plate:
            return False
        # Chặn chuỗi KHÔNG CÓ DẠNG biển số Việt Nam. Ảnh mờ, biển bị cắt mất
        # một góc hay hộp bắt nhầm vào cái cản xe vẫn ra chuỗi — nhưng không
        # bao giờ ra đúng dạng. Coi như chưa đọc được và thử lại ở khung sau,
        # còn hơn ghi một dòng rác mà người xem phải tự đoán là sai.
        if not looks_like_vn_plate(text_plate):
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

        # BỎ PHIẾU. Trước đây ghi ngay ở khung ĐẦU TIÊN đọc ra chuỗi đúng dạng,
        # nên một khung nhiễu là đủ để một dòng rác vào lịch sử vĩnh viễn — đo
        # trên 110 lượt xe thật thì 18 lượt (16%) có chuỗi đầu tiên KHÁC chuỗi
        # đa số, đúng bằng số dòng sai người dùng nhìn thấy.
        #
        # Không dùng luật cứng "phải hai khung trùng nhau": 29% số lượt chỉ đọc
        # được đúng một lần (xe đi nhanh, góc khuất) và sẽ bị mất trắng. Thay
        # vào đó CHỜ có kẻ dẫn đầu rõ ràng, và chốt lúc rời vùng nếu chưa ai đủ
        # phiếu — nên không mất lượt nào, chỉ đổi từ "chuỗi đầu tiên" sang
        # "chuỗi được nhiều khung đồng ý nhất".
        if not self._cast_vote(key, t, text_plate, subject, full_jpeg, timestamp):
            return False
        return self._persist_confirmed(
            key, meta, subject, full_jpeg, t, text_plate, timestamp, log_label,
            extra_data,
        )

    def _cast_vote(self, key, plate, text_plate, subject, full_jpeg, timestamp) -> bool:
        """Ghi một phiếu cho `plate`; True khi chuỗi này đã đủ phiếu để lưu.

        Đếm phiếu theo chuỗi ĐÃ LÀM SẠCH (`plate`) nhưng cất kèm chuỗi HIỂN THỊ
        (`text_plate`, còn dấu gạch giữa hai dòng) — cột plate_number lưu dạng
        hiển thị, làm sạch chỉ để so sánh.

        Cũng giữ lại ẢNH VÀ HỘP của ứng viên đang dẫn đầu, vì lúc xe rời vùng
        thì detection không còn trong meta nữa — không cất sẵn thì
        `_flush_leader` chẳng có gì để lưu."""
        votes = self._votes.setdefault(key, {})
        votes[plate] = votes.get(plate, 0) + 1
        leader = self._leader.get(key)
        # Làm mới ảnh của chính ứng viên dẫn đầu (ảnh mới thường rõ hơn vì xe
        # đang tiến lại gần), và đổi ngôi khi có kẻ VƯỢT HẲN — hoà thì giữ
        # nguyên kẻ đang giữ ngôi, tức chuỗi đọc được sớm hơn.
        if leader is None or plate == leader[0] or votes[plate] > votes.get(leader[0], 0):
            self._leader[key] = (plate, text_plate, subject, full_jpeg, timestamp)
        return votes[plate] >= self._VOTES_TO_CONFIRM

    def _flush_leader(self, key, meta, timestamp, extra_data=None) -> None:
        """Chốt ứng viên dẫn đầu khi xe rời vùng mà chưa ai đủ phiếu.

        Nhờ bước này việc bỏ phiếu KHÔNG làm mất lượt xe nào: xe chỉ đọc được
        đúng một khung vẫn được ghi, chỉ là ghi muộn hơn vài trăm mili giây."""
        if self._track_state.get(key) != _PENDING:
            return
        leader = self._leader.get(key)
        if leader is None:
            return
        plate, text_plate, subject, full_jpeg, ts = leader
        self._persist_confirmed(
            key, meta, subject, full_jpeg, plate, text_plate, ts, "exited_zone",
            extra_data,
        )

    def _persist_confirmed(self, key, meta, subject, full_jpeg, plate, text_plate,
                           timestamp, log_label, extra_data=None) -> bool:
        # Chống trùng khi ra-vào lại: cùng một tracker có thể rời vùng trong
        # chốc lát (hoặc mất detection) làm _track_state bị xoá, rồi vào lại
        # với đúng id cũ. Bỏ qua dòng trùng nhưng vẫn đánh dấu RESOLVED để
        # in_the_area ngừng thử lại.
        last = self._last_saved.get(key)
        if last is not None and timestamp - last < self._REENTER_COOLDOWN_S:
            self._track_state[key] = _RESOLVED
            return True
        # Chống trùng theo CHÍNH CHUỖI BIỂN, không chỉ theo tracker id. Cùng
        # một chiếc xe vẫn có thể mang hai tracker id: model vẽ hai hộp chồng
        # nhau lên nó, hoặc tracker đánh mất rồi cấp id mới khi xe bị che. Lúc
        # đó khoá theo id không chặn được gì và lịch sử có hai dòng y hệt nhau
        # (đo thật: '61A-27823' hai lần cách nhau vài giây). Hai xe khác nhau
        # cùng biển trong 30 giây là chuyện không có thật, nên chặn ở đây an toàn.
        plate_key = (str(meta["cameraId"]), plate.replace(" ", ""))
        last_plate = self._last_saved_plate.get(plate_key)
        if last_plate is not None and timestamp - last_plate < self._REENTER_COOLDOWN_S:
            self._track_state[key] = _RESOLVED
            return True
        self._last_saved_plate[plate_key] = timestamp
        # Đã xác nhận — đổi trạng thái ĐỒNG BỘ trước khi chạy tác vụ lưu bất
        # đồng bộ, để in_the_area của frame ngay sau đó thấy _RESOLVED và
        # không lên lịch lưu trùng.
        self._last_saved[key] = timestamp
        self._track_state[key] = _RESOLVED
        # Nhả ảnh đang giữ cho ứng viên: mỗi tracker ôm một khung JPEG toàn cảnh,
        # để lại thì camera đông xe là phình bộ nhớ vô ích.
        self._votes.pop(key, None)
        self._leader.pop(key, None)
        print(f"{log_label} id={key[1]} plate={text_plate}")
        asyncio.create_task(
            self._persist_event(meta, subject, full_jpeg, text_plate, timestamp,
                                extra_data)
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
        # Lượt mới thì phiếu cũ của đúng id đó (tracker tái sử dụng số) không
        # được phép cộng dồn sang.
        self._votes.pop(key, None)
        self._leader.pop(key, None)
        self._try_confirm_plate(
            meta, parent, full_jpeg, timestamp, secondary_conf,
            key, "entered_zone", extra_data=extra_data,
        )

    def dwell_alert(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        print(f"stayed_zone")

    def exited_zone(self, id, meta, full_jpeg, timestamp, secondary_conf, extra_data=None, zone_idx=0):
        key = (str(meta["cameraId"]), int(id))
        # Chốt ứng viên dẫn đầu TRƯỚC khi xoá trạng thái: xe đọc được đúng một
        # khung vẫn phải được ghi, chỉ là ghi ở đây thay vì ngay lúc đọc.
        self._flush_leader(key, meta, timestamp, extra_data)
        self._track_state.pop(key, None)
        self._votes.pop(key, None)
        self._leader.pop(key, None)
        # Giữ lại các mốc thời gian lưu gần đây phục vụ cooldown ra-vào lại;
        # bỏ những mốc đã quá hạn để dict không phình vô hạn.
        cutoff = timestamp - self._REENTER_COOLDOWN_S
        self._last_saved = {k: t for k, t in self._last_saved.items() if t >= cutoff}
        self._last_saved_plate = {
            k: t for k, t in self._last_saved_plate.items() if t >= cutoff
        }
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
