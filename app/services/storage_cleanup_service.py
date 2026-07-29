# -*- coding: utf-8 -*-
"""Dọn dung lượng theo mô hình GIỮ TỐI THIỂU N GB TRỐNG (xem storage_policy.py).

Chạy định kỳ (task_storage_cleanup): mỗi lần đo chỗ trống THẬT của ổ; nếu trống
< min_free_gb thì xoá dữ liệu CŨ NHẤT của 5 loại — ưu tiên loại đang vượt phần
được-giữ theo trọng số — cho tới khi trống ≥ target_free_gb hoặc hết dữ liệu.

Xoá cả FILE trên đĩa lẫn HÀNG trong DB. Recording đang ghi dở (status khác
'complete' hoặc mới < 3 phút) TUYỆT ĐỐI không đụng tới.
"""

import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ai_service_base import UPLOADS_ROOT

GB = 1024 ** 3
_BATCH = 200  # số hàng xoá mỗi nhịp trước khi đo lại chỗ trống

# gstreamer_c (chứa recordings/) là thư mục anh em với repo python.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GSTREAMER_C_DIR = os.environ.get(
    "STORAGE_GSTREAMER_C_DIR",
    os.path.join(os.path.dirname(_REPO_DIR), "gstreamer_c"),
)
RECORDINGS_DIR = os.path.join(GSTREAMER_C_DIR, "recordings")


@dataclass
class Category:
    key: str
    weight_field: str          # cột trọng số trong storage_policy
    directory: str             # thư mục để đo dung lượng (du)
    table: str                 # bảng để xoá hàng
    order_col: str             # cột sắp theo CŨ->MỚI
    file_cols: List[str]       # (các) cột chứa đường dẫn/URL file
    file_kind: str             # "uploads" | "recording"
    where: str = ""            # điều kiện WHERE thêm (an toàn)


CATEGORIES: List[Category] = [
    Category(
        key="record", weight_field="w_record",
        directory=RECORDINGS_DIR, table="recording_segments",
        order_col="start_at", file_cols=["path"], file_kind="recording",
        # Không đụng đoạn đang ghi / vừa đóng chưa chắc chắn.
        where="status = 'complete' AND start_at < now() - interval '3 minutes'",
    ),
    Category(
        key="event_face", weight_field="w_event_face",
        directory=os.path.join(UPLOADS_ROOT, "faces"), table="event_face",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads",
    ),
    Category(
        key="event_plate", weight_field="w_event_plate",
        directory=os.path.join(UPLOADS_ROOT, "plates"), table="event_plates",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads",
    ),
    Category(
        key="parking_lot_event", weight_field="w_parking_lot_event",
        directory=os.path.join(UPLOADS_ROOT, "parking"), table="parking_lot_event",
        order_col="timestamp", file_cols=["face_image_full", "plate_image_full"],
        file_kind="uploads",
    ),
    Category(
        key="restricted_area", weight_field="w_restricted_area",
        directory=os.path.join(UPLOADS_ROOT, "restricted"), table="restricted_areas",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads",
    ),
]


def _du_bytes(path: str) -> int:
    """Dung lượng thư mục (bytes). du nhanh hơn os.walk cho nhiều file."""
    if not os.path.isdir(path):
        return 0
    try:
        out = subprocess.run(
            ["du", "-sb", path], capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            return int(out.stdout.split()[0])
    except Exception:
        pass
    # Dự phòng: walk cộng dồn.
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _abs_path(kind: str, value: str) -> str:
    """Đổi giá trị lưu trong DB -> đường dẫn tuyệt đối trên đĩa."""
    if not value:
        return ""
    if kind == "recording":
        # value = "recordings/<id>/<file>.ts" (tương đối gstreamer_c)
        return os.path.join(GSTREAMER_C_DIR, value)
    # uploads: value = "/uploads/<...>"
    rel = value[len("/uploads/"):] if value.startswith("/uploads/") else value.lstrip("/")
    return os.path.join(UPLOADS_ROOT, rel)


@dataclass
class _RunStats:
    freed_bytes: int = 0
    deleted_rows: int = 0
    per_category: dict = field(default_factory=dict)


class StorageCleanupService:
    async def _load_policy(self, session: AsyncSession):
        row = (await session.execute(text(
            "SELECT enabled, min_free_gb, target_free_gb, "
            "w_record, w_event_face, w_event_plate, "
            "w_parking_lot_event, w_restricted_area "
            "FROM storage_policy WHERE id = 1"
        ))).mappings().first()
        return row

    def _weights(self, policy) -> dict:
        raw = {c.key: max(0.0, float(policy[c.weight_field])) for c in CATEGORIES}
        s = sum(raw.values())
        if s <= 0:
            # Tất cả 0 -> chia đều để vẫn xoá được.
            return {k: 1.0 / len(raw) for k in raw}
        return {k: v / s for k, v in raw.items()}

    async def _delete_batch(self, session: AsyncSession, cat: Category, n: int):
        """Xoá n hàng CŨ NHẤT của một loại (kèm file). Trả (bytes, số hàng)."""
        where = f"WHERE {cat.where}" if cat.where else ""
        cols = ", ".join(cat.file_cols)
        rows = (await session.execute(text(
            f"SELECT CAST(id AS text) AS id, {cols} FROM {cat.table} "
            f"{where} ORDER BY {cat.order_col} ASC LIMIT :n"
        ), {"n": n})).mappings().all()
        if not rows:
            return 0, 0

        freed = 0
        ids = []
        for r in rows:
            ids.append(r["id"])
            for col in cat.file_cols:
                p = _abs_path(cat.file_kind, r[col])
                if not p:
                    continue
                try:
                    freed += os.path.getsize(p)
                    os.remove(p)
                except OSError:
                    pass  # file đã mất / không đọc được: bỏ qua, vẫn xoá hàng

        await session.execute(
            text(f"DELETE FROM {cat.table} WHERE CAST(id AS text) = ANY(:ids)"),
            {"ids": ids},
        )
        await session.commit()
        return freed, len(ids)

    async def run_once(self, session: AsyncSession) -> _RunStats:
        stats = _RunStats()
        policy = await self._load_policy(session)
        if not policy or not policy["enabled"]:
            return stats

        min_free = float(policy["min_free_gb"]) * GB
        target_free = max(float(policy["target_free_gb"]) * GB, min_free)

        usage = shutil.disk_usage(RECORDINGS_DIR if os.path.isdir(RECORDINGS_DIR) else "/")
        if usage.free >= min_free:
            return stats  # còn đủ trống, không làm gì

        weights = self._weights(policy)
        sizes = {c.key: _du_bytes(c.directory) for c in CATEGORIES}
        exhausted = set()  # loại đã hết hàng để xoá

        guard = 0
        while guard < 5000:
            guard += 1
            usage = shutil.disk_usage(RECORDINGS_DIR if os.path.isdir(RECORDINGS_DIR) else "/")
            if usage.free >= target_free:
                break  # đã đủ chỗ trống -> DỪNG (chỉ giải phóng đúng lượng cần)

            # Chọn loại "NẶNG CÂN NHẤT" = size / trọng-số lớn nhất, trong các
            # loại còn hàng. Trọng số cao được giữ nhiều hơn (mẫu số lớn -> ít bị
            # chọn). Loại rỗng bị loại khỏi tranh chọn nên phần của nó tự nhường
            # cho loại còn dữ liệu — record không bị ép về 60% khi loại khác rỗng.
            best = None
            best_ratio = -1.0
            for c in CATEGORIES:
                if c.key in exhausted or sizes[c.key] <= 0:
                    continue
                ratio = sizes[c.key] / max(weights[c.key], 1e-9)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = c
            if best is None:
                break  # không còn gì để xoá ở đâu cả (đã dọn hết dữ liệu của mình)

            freed, ndel = await self._delete_batch(session, best, _BATCH)
            if ndel == 0:
                # Bảng hết hàng dù thư mục còn dung lượng (file mồ côi) -> ngừng
                # loại này để khỏi lặp vô ích.
                exhausted.add(best.key)
                sizes[best.key] = 0
                continue
            sizes[best.key] = max(0, sizes[best.key] - freed)
            stats.freed_bytes += freed
            stats.deleted_rows += ndel
            stats.per_category[best.key] = stats.per_category.get(best.key, 0) + freed

        return stats

    async def status(self, session: AsyncSession) -> dict:
        """Ảnh chụp cho API/UI: đĩa + kích thước từng loại + chính sách."""
        policy = await self._load_policy(session)
        usage = shutil.disk_usage(RECORDINGS_DIR if os.path.isdir(RECORDINGS_DIR) else "/")
        sizes = {c.key: _du_bytes(c.directory) for c in CATEGORIES}
        return {
            "disk": {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "used_percent": round(usage.used / usage.total * 100, 1),
                "free_gb": round(usage.free / GB, 2),
            },
            "categories": {
                k: {"size_bytes": v, "size_gb": round(v / GB, 3)}
                for k, v in sizes.items()
            },
            "policy": dict(policy) if policy else None,
        }


storage_cleanup_service = StorageCleanupService()
