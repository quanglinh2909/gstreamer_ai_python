import re
import threading
from typing import Dict, List, Optional
import Levenshtein
import httpx
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import time
from app.utils.open_door.door_manager import door_manager

from app.models.plate_white_list import PlateWhiteList
from app.repositories.plate_white_list_repository import PlateWhiteListRepository
from app.services.plate_white_list_settings_service import (
    plate_white_list_settings_service,
)
from app.utils.plate_recognition_hepper import detect_plate_from_children


def _normalize(plate_number: str) -> str:
    # Keep only letters and digits, upper-cased — drop spaces, dots, dashes
    # and every other symbol so every lookup path (is_whitelisted, the cache
    # keys and process_ai_result) compares plates the same canonical way and
    # doesn't miss because of trivial OCR formatting differences.
    return re.sub(r"[^A-Za-z0-9]", "", plate_number or "").upper()


def _clean_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    trimmed = name.strip()
    return trimmed or None


class PlateWhiteListService:
    def __init__(self):
        # plate_number (normalised, UPPER) -> {"last_matched": {scope: ts}}.
        # `scope` = "group:<id>" nếu camera thuộc CỤM CỔNG, ngược lại
        # "cam:<camera_id>" (PlateGateSettings.cooldown_scope). Camera đứng
        # riêng vẫn có đồng hồ riêng như trước; các camera cùng cụm chia nhau
        # một đồng hồ.
        # The ALPR hot path only needs membership ("is this plate
        # whitelisted?"), so a plate that has never opened a barrier keeps an
        # empty dict; process_ai_result fills in the per-camera timestamps it
        # uses for rate limiting. Lookups stay O(1).
        self.plate_white_list: Dict[str, Dict] = {}
        # Cross-thread access: API handlers (FastAPI loop) mutate; the
        # process_ai_service thread reads on every detection.
        self._cache_lock = threading.RLock()
        # camera_id đã in cảnh báo "chưa cấu hình". Bỏ qua trong im lặng thì
        # rất khó hiểu tại sao cổng không mở, nhưng in mỗi frame thì ngập log
        # — nên in đúng một lần cho mỗi camera.
        self._warned_unconfigured = set()

    def _warn_unconfigured(self, camera_id: str) -> None:
        with self._cache_lock:
            if camera_id in self._warned_unconfigured:
                return
            self._warned_unconfigured.add(camera_id)
        print(
            f"[plate whitelist] camera {camera_id} chua co "
            f"plate_white_list_settings -> bo qua whitelist/barrier. "
            f"Cau hinh: PUT /plate-white-list-settings/{camera_id}"
        )

    async def load_all(self, db: AsyncSession) -> None:
        """Prime the in-memory cache from DB. Call once at app startup
        (after Base.metadata.create_all) so the first lookup doesn't have
        to wait on I/O."""
        rows = await PlateWhiteListRepository.list_all(db)
        with self._cache_lock:
            self.plate_white_list = {_normalize(row.plate_number): {} for row in rows}
        print(f"[plate whitelist] loaded {len(self.plate_white_list)} entries")

    def is_whitelisted(self, plate_number: str) -> bool:
        """Thread-safe O(1) membership check for the ALPR pipeline."""
        normalized = _normalize(plate_number)
        with self._cache_lock:
            return normalized in self.plate_white_list

    def all_cached(self) -> List[str]:
        with self._cache_lock:
            return list(self.plate_white_list.keys())

    async def create(
        self, db: AsyncSession, plate_number: str, name: Optional[str] = None,
    ) -> PlateWhiteList:
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        existing = await PlateWhiteListRepository.get_by_plate(db, normalized)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Plate {normalized} already in whitelist",
            )
        entry = await PlateWhiteListRepository.create(db, normalized, _clean_name(name))
        with self._cache_lock:
            self.plate_white_list[entry.plate_number] = {}
        return entry

    async def update(
        self,
        db: AsyncSession,
        entry_id: int,
        plate_number: str,
        name: Optional[str] = None,
    ) -> PlateWhiteList:
        entry = await PlateWhiteListRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Whitelist entry not found")
        normalized = _normalize(plate_number)
        if not normalized:
            raise HTTPException(status_code=400, detail="Plate number is empty")
        # Only check uniqueness when the value actually changes — otherwise a
        # self-update would falsely conflict with its own row.
        if normalized != entry.plate_number:
            existing = await PlateWhiteListRepository.get_by_plate(db, normalized)
            if existing is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"Plate {normalized} already in whitelist",
                )
        old_plate = entry.plate_number
        entry = await PlateWhiteListRepository.update(
            db, entry, normalized, _clean_name(name),
        )
        with self._cache_lock:
            # Drop the old key first in case the plate number itself changed,
            # otherwise the rename would leave a stale duplicate in the cache.
            if old_plate != entry.plate_number:
                self.plate_white_list.pop(old_plate, None)
            self.plate_white_list[entry.plate_number] = {}
        return entry

    async def get(self, db: AsyncSession, entry_id: int) -> Optional[PlateWhiteList]:
        return await PlateWhiteListRepository.get(db, entry_id)

    async def list_paginated(
        self,
        db: AsyncSession,
        page: int,
        size: int,
        plate_number: Optional[str] = None,
    ):
        return await PlateWhiteListRepository.list_paginated(
            db, page, size, plate_number,
        )

    async def delete(self, db: AsyncSession, entry_id: int) -> None:
        entry = await PlateWhiteListRepository.get(db, entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Whitelist entry not found")
        plate = entry.plate_number  # capture before SQLAlchemy invalidates the row
        await PlateWhiteListRepository.delete(db, entry)
        with self._cache_lock:
            self.plate_white_list.pop(plate, None)
    
    async def process_ai_result(self, children, camera_id: str):
        """Đọc biển từ các OCR children rồi đối chiếu whitelist và mở barrier.

        Nhánh whitelist TỰ đọc biển bằng ngưỡng ocr_confidence của chính
        camera đó, không dùng lại chuỗi biển mà plate_recognition_service đã
        dựng bằng secondaryConf của AI job. Hai nhánh phục vụ hai mục đích
        khác nhau — bên kia lưu EventPlate để người xem lại, bên này quyết
        định có mở cổng hay không — nên siết ngưỡng cho cổng không kéo theo
        thay đổi dữ liệu sự kiện, và ngược lại.

        Mọi ngưỡng đều lấy từ PlateWhiteListSettings của camera (đọc từ cache
        trong RAM, không truy vấn DB) nên gọi hàm này ở mọi frame là an toàn.
        Camera chưa có dòng cấu hình thì KHÔNG chạy gì cả.
        """
        camera_id = str(camera_id)
        cfg = plate_white_list_settings_service.get_cached(camera_id)
        if cfg is None:
            self._warn_unconfigured(camera_id)
            return

        # Ký tự yếu hơn ocr_confidence bị LOẠI khỏi chuỗi (không phải chặn cả
        # lần đọc), nên biển đọc ra ngắn đi và thường rớt ngay ở
        # min_plate_length bên dưới.
        plate_number = _normalize(
            detect_plate_from_children(children, cfg.ocr_confidence)
        )

        # Đọc thiếu ký tự thì bỏ qua ngay — rẻ hơn nhiều so với quét toàn bộ
        # whitelist, và đây là nguyên nhân chính gây mở nhầm cổng.
        if len(plate_number) < cfg.min_plate_length:
            return

        now = time.time()
        # PHẠM VI và THỜI GIAN của đồng hồ chờ — cả hai đã được giải sẵn lúc
        # nạp cache (plate_white_list_settings_service._build_cache), nên ở
        # đây không có nhánh if nào về cụm cả:
        #
        #   thuộc cụm  -> scope "group:<id>", chờ theo pre_time của CỤM
        #   đứng riêng -> scope "cam:<id>",   chờ theo pre_time của CAMERA
        #
        # Mặc định tách theo camera là CỐ Ý: cổng vào và cổng ra là hai camera
        # khác nhau, xe vừa vào không được vì thế mà bị khoá ở cổng ra. Cụm
        # dành cho làn vừa vào vừa ra, nơi hai camera cùng điều khiển MỘT
        # barrier — xe qua camera 1 mở cổng, chạy tiếp qua camera 2 lại mở lần
        # nữa nên barrier không kịp đóng.
        scope = cfg.cooldown_scope
        # Snapshot under the lock: API handlers (FastAPI thread) mutate this
        # dict via create/update/delete while we run on the recv-loop thread,
        # so iterating it directly risks "dict changed size during iteration".
        with self._cache_lock:
            snapshot = [
                (key, meta.get("last_matched", {}).get(scope, 0.0))
                for key, meta in self.plate_white_list.items()
            ]
        # Chọn biển GẦN NHẤT rồi mới xét thời gian chờ, chứ không lấy biển
        # đầu tiên nằm dưới ngưỡng. Với max_edit_distance > 0, một chuỗi đọc
        # được có thể khớp nhiều biển cùng lúc (biển VN hay chỉ khác nhau 1-2
        # ký tự); lấy biển đầu tiên gặp thì thứ tự dict quyết định xe nào
        # được ghi nhận, và tệ hơn: khi biển đúng đang trong thời gian chờ,
        # một biển gần giống sẽ lọt qua và mở cổng thay nó.
        #
        # Key trong cache đã được _normalize lúc create/update/load_all nên
        # không chuẩn hoá lại ở đây — vòng lặp này chạy mỗi frame.
        best_key = None
        best_distance = None
        best_last_matched = 0.0
        for key, last_matched in snapshot:
            distance = Levenshtein.distance(plate_number, key)
            if distance > cfg.max_edit_distance:
                continue
            # Hoà thì lấy key nhỏ hơn theo thứ tự chữ cái, để cùng một chuỗi
            # đọc được luôn cho ra cùng một kết quả bất kể thứ tự dict.
            if (
                best_distance is None
                or distance < best_distance
                or (distance == best_distance and key < best_key)
            ):
                best_key = key
                best_distance = distance
                best_last_matched = last_matched
            if distance == 0:
                break  # khớp tuyệt đối, không thể có kết quả tốt hơn

        if best_key is None:
            return
        if best_last_matched:
            if cfg.cooldown_seconds <= 0:
                # 0 = không cho mở lại: biển này đã mở một lần rồi.
                return
            if now - best_last_matched <= cfg.cooldown_seconds:
                return

        # Re-check + stamp under the lock; the entry may have been removed by
        # a concurrent delete since the snapshot, and the stamp also rate-
        # limits the next match (cooldown_seconds giây).
        with self._cache_lock:
            entry = self.plate_white_list.get(best_key)
            if entry is None:
                return
            entry.setdefault("last_matched", {})[scope] = now
        print(
            f"Plate {plate_number} matched whitelist entry {best_key} "
            f"with distance {best_distance} (camera {camera_id}"
            # In cả cụm và số giây THỰC SỰ áp dụng: khi cổng không mở, câu hỏi
            # đầu tiên luôn là "bị chặn bởi chính camera này hay bởi camera
            # khác cùng cụm, và đang tính theo mốc nào".
            + (f", cum '{cfg.gate_group_name}'" if cfg.gate_group_name else "")
            + f", cho {cfg.cooldown_seconds}s)"
        )
        try:
            door_manager.open_door(cfg.barrier_duration)
            print(f"Barrier opened when plate {plate_number} matched whitelist entry {best_key}")
        except Exception as e:
            print(f"Failed to open barrier for plate {plate_number} matched whitelist entry {best_key}: {e}")




plate_white_list_service = PlateWhiteListService()
