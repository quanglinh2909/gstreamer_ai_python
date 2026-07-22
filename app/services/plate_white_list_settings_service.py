import threading
from dataclasses import dataclass
from typing import Dict, Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plate_white_list_settings import PlateWhiteListSettings
from app.repositories.plate_white_list_settings_repository import (
    PlateWhiteListSettingsRepository,
)


@dataclass(frozen=True)
class PlateGateSettings:
    """Ảnh chụp bất biến của cấu hình một camera.

    Frozen để luồng đọc (recv-loop của process_ai_service) lấy được nguyên
    một bộ giá trị nhất quán chỉ bằng một lần đọc dict, không sợ đọc phải
    trạng thái nửa vời khi API đang ghi. Muốn đổi giá trị thì thay cả object
    chứ không sửa tại chỗ.
    """

    pre_time: int
    max_edit_distance: int
    ocr_confidence: float
    min_plate_length: int
    barrier_duration: float


def _to_value(row: PlateWhiteListSettings) -> PlateGateSettings:
    return PlateGateSettings(
        pre_time=int(row.pre_time),
        max_edit_distance=int(row.max_edit_distance),
        ocr_confidence=float(row.ocr_confidence),
        min_plate_length=int(row.min_plate_length),
        barrier_duration=float(row.barrier_duration),
    )


class PlateWhiteListSettingsService:
    """Cache cấu hình whitelist theo camera.

    Mỗi biển đọc được đều phải tra cấu hình, tức vài lần mỗi giây cho mỗi
    camera. Truy vấn DB ở đó sẽ chặn recv-loop, nên toàn bộ bảng được nạp vào
    RAM lúc khởi động và mọi thao tác ghi qua API cập nhật cache ngay trong
    cùng hàm — DB chỉ còn là nơi lưu bền, không nằm trên đường nóng.
    """

    def __init__(self):
        # camera_id -> PlateGateSettings
        self._cache: Dict[str, PlateGateSettings] = {}
        # API handler (luồng FastAPI) ghi, recv-loop đọc → phải khoá.
        self._lock = threading.RLock()

    async def load_all(self, db: AsyncSession) -> None:
        """Nạp cache từ DB. Gọi một lần lúc khởi động, SAU create_all và
        TRƯỚC khi luồng process_ai_service chạy."""
        rows = await PlateWhiteListSettingsRepository.list_all(db)
        cache = {str(row.camera_id): _to_value(row) for row in rows}
        with self._lock:
            self._cache = cache
        print(f"[plate whitelist settings] loaded {len(cache)} cameras")

    def get_cached(self, camera_id: str) -> Optional[PlateGateSettings]:
        """Tra cứu O(1), an toàn đa luồng — dùng trên đường nóng.

        Trả về None khi camera CHƯA được cấu hình. Không có giá trị mặc định
        nào thay thế: mở barrier là hành động vật lý, nên phải do người dùng
        bật lên một cách rõ ràng cho từng camera chứ không tự chạy vì quên
        cấu hình. Bên gọi phải coi None là "bỏ qua".
        """
        with self._lock:
            return self._cache.get(str(camera_id))

    def is_configured(self, camera_id: str) -> bool:
        with self._lock:
            return str(camera_id) in self._cache

    def all_cached(self) -> Dict[str, PlateGateSettings]:
        with self._lock:
            return dict(self._cache)

    async def get(
        self, db: AsyncSession, camera_id: str,
    ) -> Optional[PlateWhiteListSettings]:
        return await PlateWhiteListSettingsRepository.get_by_camera(db, camera_id)

    async def list_all(self, db: AsyncSession):
        return await PlateWhiteListSettingsRepository.list_all(db)

    async def upsert(
        self, db: AsyncSession, camera_id: str, settings: Dict,
    ) -> PlateWhiteListSettings:
        camera_id = (camera_id or "").strip()
        if not camera_id:
            raise HTTPException(status_code=400, detail="camera_id is empty")
        entry = await PlateWhiteListSettingsRepository.upsert(
            db, camera_id, settings,
        )
        with self._lock:
            self._cache[str(entry.camera_id)] = _to_value(entry)
        return entry

    async def delete(self, db: AsyncSession, camera_id: str) -> None:
        """Xoá cấu hình → TẮT hẳn nhánh whitelist/barrier của camera này."""
        entry = await PlateWhiteListSettingsRepository.get_by_camera(db, camera_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Settings not found")
        await PlateWhiteListSettingsRepository.delete(db, entry)
        with self._lock:
            self._cache.pop(str(camera_id), None)


plate_white_list_settings_service = PlateWhiteListSettingsService()
