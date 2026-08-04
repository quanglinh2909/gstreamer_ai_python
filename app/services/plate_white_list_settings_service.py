import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plate_white_list_settings import PlateWhiteListSettings
from app.repositories.plate_gate_group_repository import PlateGateGroupRepository
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

    # --- Ba trường dưới đây được GIẢI SẴN lúc nạp cache, không lưu trong DB ---
    #
    # Đường nóng chạy mỗi khung hình nên nó không được phép tra bảng cụm rồi
    # tự quyết "lấy pre_time của ai". Quy tắc gộp cụm nằm ở ĐÚNG MỘT chỗ (hàm
    # _build_cache) thay vì nằm rải trong logic mở cổng.

    # Khoá của đồng hồ chờ: "group:<id>" nếu thuộc cụm, "cam:<camera_id>" nếu
    # đứng riêng. Hai camera cùng cụm ra cùng một khoá — đó là toàn bộ cơ chế.
    cooldown_scope: str
    # Số giây chờ THỰC SỰ áp dụng: của cụm nếu thuộc cụm, của camera nếu không.
    # Không phải max, không phải min, không cộng dồn — xem plate_gate_group.py.
    cooldown_seconds: int
    # Tên cụm để in log và hiện lên giao diện. "" = không thuộc cụm nào.
    gate_group_name: str


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
        # group_id -> (name, pre_time). Giữ để API trả về và để dựng lại cache.
        self._groups: Dict[int, tuple] = {}
        # API handler (luồng FastAPI) ghi, recv-loop đọc → phải khoá.
        self._lock = threading.RLock()

    def _build_cache(self, rows, groups: Dict[int, tuple]) -> Dict[str, PlateGateSettings]:
        """Gộp cấu hình camera với cụm của nó. ĐÂY là chỗ duy nhất định nghĩa
        "thuộc cụm thì chờ bao lâu"."""
        cache = {}
        for row in rows:
            gid = row.gate_group_id
            group = groups.get(gid) if gid is not None else None
            if group is None:
                # Không thuộc cụm (hoặc cụm đã bị xoá) -> hành vi cũ, từng bit.
                scope = f"cam:{row.camera_id}"
                seconds = int(row.pre_time)
                name = ""
            else:
                name, group_pre_time = group
                scope = f"group:{gid}"
                seconds = int(group_pre_time)
            cache[str(row.camera_id)] = PlateGateSettings(
                pre_time=int(row.pre_time),
                max_edit_distance=int(row.max_edit_distance),
                ocr_confidence=float(row.ocr_confidence),
                min_plate_length=int(row.min_plate_length),
                barrier_duration=float(row.barrier_duration),
                cooldown_scope=scope,
                cooldown_seconds=seconds,
                gate_group_name=name,
            )
        return cache

    async def load_all(self, db: AsyncSession) -> None:
        """Nạp lại TOÀN BỘ cache từ DB (camera + cụm).

        Gọi lúc khởi động và sau MỌI thay đổi cụm. Nạp lại cả bảng thay vì sửa
        từng khoá vì một thao tác trên cụm đổi cấu hình của nhiều camera cùng
        lúc — mà bảng này chỉ lớn bằng số camera, nạp lại là chuyện vài mili
        giây.
        """
        rows = await PlateWhiteListSettingsRepository.list_all(db)
        groups = {
            g.id: (g.name, int(g.pre_time))
            for g in await PlateGateGroupRepository.list_all(db)
        }
        cache = self._build_cache(rows, groups)
        with self._lock:
            self._groups = groups
            self._cache = cache
        print(
            f"[plate whitelist settings] loaded {len(cache)} cameras, "
            f"{len(groups)} cum cong"
        )

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
        gid = settings.get("gate_group_id")
        if gid is not None:
            # Gán vào cụm không tồn tại thì camera sẽ lặng lẽ rơi về chế độ
            # đứng riêng — im lặng đúng ở chỗ nguy hiểm nhất, nên chặn ngay.
            if await PlateGateGroupRepository.get_by_id(db, int(gid)) is None:
                raise HTTPException(status_code=400, detail=f"Cum {gid} khong ton tai")
        entry = await PlateWhiteListSettingsRepository.upsert(
            db, camera_id, settings,
        )
        await self.load_all(db)
        return entry

    async def delete(self, db: AsyncSession, camera_id: str) -> None:
        """Xoá cấu hình → TẮT hẳn nhánh whitelist/barrier của camera này."""
        entry = await PlateWhiteListSettingsRepository.get_by_camera(db, camera_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="Settings not found")
        await PlateWhiteListSettingsRepository.delete(db, entry)
        with self._lock:
            self._cache.pop(str(camera_id), None)

    # ------------------------- Cụm cổng -------------------------

    def groups_cached(self) -> Dict[int, tuple]:
        with self._lock:
            return dict(self._groups)

    def cameras_in_group(self, group_id: int) -> List[str]:
        scope = f"group:{group_id}"
        with self._lock:
            return [
                cam for cam, cfg in self._cache.items()
                if cfg.cooldown_scope == scope
            ]


plate_white_list_settings_service = PlateWhiteListSettingsService()
