# -*- coding: utf-8 -*-
"""Dọn dung lượng. HAI luật chạy song song, mỗi luật trả lời một câu hỏi khác:

1. HẠN LƯU THEO NGÀY, riêng từng camera (`cameras.retention_days`) — chạy MỌI
   chu kỳ, không cần đĩa đầy. "Camera này chỉ giữ 7 ngày" là một lời hứa với
   người dùng, không phải phương án chữa cháy: ổ 8TB không bao giờ đầy thì
   luật (2) không bao giờ chạy và dữ liệu nằm lại vĩnh viễn.
   Vì luật này chỉ đọc DB nên nó vẫn dọn được camera ĐÃ TẮT GHI HÌNH, thậm chí
   camera đã ngắt kết nối — thứ mà một luật gắn vào luồng ghi không làm được.

2. GIỮ TỐI THIỂU N GB TRỐNG (xem storage_policy.py) — chỉ chạy khi chỗ trống
   THẬT của ổ tụt dưới min_free_gb, rồi xoá dữ liệu CŨ NHẤT của 7 loại (ưu tiên
   loại đang vượt phần được-giữ theo trọng số) cho tới khi trống ≥
   target_free_gb hoặc hết dữ liệu.

Hai luật KHÔNG mâu thuẫn nhau vì cả hai chỉ biết xoá: (1) đặt trần TUỔI, (2)
đặt trần DUNG LƯỢNG. Đĩa đầy thì (2) vẫn cắt vào dữ liệu còn trong hạn — tức là
hạn lưu theo ngày là mức TỐI ĐA được giữ, không phải mức tối thiểu được đảm bảo.

Xoá cả FILE trên đĩa lẫn HÀNG trong DB. Recording đang ghi dở (status khác
'complete' hoặc mới < 3 phút) TUYỆT ĐỐI không đụng tới.

Chuyển động và khẩu trang giờ cũng nằm trong danh sách này. Trước đây chúng
đứng ngoài vì mỗi loại thiếu một nửa: khẩu trang không có bảng, chuyển động
không có file — nên chuyển động phải dọn bằng một luật riêng ("đoạn ghi chứa nó
đã bị xoá thì xoá theo"). Luật đó gắn tuổi thọ của sự kiện vào tuổi thọ của
video, tức là người dùng không hề chỉnh được nó ở đâu. Giờ cả hai đều có bảng
lẫn ảnh, nên chúng dọn theo TỶ TRỌNG như mọi loại khác.
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
# Ảnh khung hình của sự kiện chuyển động, do engine C++ ghi. Nằm cạnh
# recordings/ chứ không nằm trong (xem StreamTypes.hpp::motionSnapshotDir), để
# `du` đo được tách bạch hai loại.
MOTION_SNAPSHOTS_DIR = os.path.join(GSTREAMER_C_DIR, "motion-snapshots")


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
    # Kiểu của order_col, để tính TUỔI cho hạn-lưu-theo-ngày. Hai backend viết
    # thời gian theo hai kiểu khác nhau và không có ý định gộp: engine C++ dùng
    # timestamptz, còn các bảng sự kiện của Python dùng epoch giây (integer).
    time_kind: str = "epoch"   # "epoch" | "tstz"
    # (Các) cột chỉ tới camera sở hữu hàng. Rỗng = loại này đứng ngoài hạn lưu
    # theo camera. Nhiều cột (bãi xe: mặt + biển) thì hàng chỉ hết hạn khi ĐÃ
    # quá hạn của MỌI camera liên quan — xem _retention_clauses.
    camera_cols: List[str] = field(default_factory=list)


CATEGORIES: List[Category] = [
    Category(
        key="record", weight_field="w_record",
        directory=RECORDINGS_DIR, table="recording_segments",
        order_col="start_at", file_cols=["path"], file_kind="recording",
        # Không đụng đoạn đang ghi / vừa đóng chưa chắc chắn.
        where="status = 'complete' AND start_at < now() - interval '3 minutes'",
        time_kind="tstz", camera_cols=["camera_id"],
    ),
    Category(
        key="event_face", weight_field="w_event_face",
        directory=os.path.join(UPLOADS_ROOT, "faces"), table="event_face",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads", camera_cols=["camera_id"],
    ),
    Category(
        key="event_plate", weight_field="w_event_plate",
        directory=os.path.join(UPLOADS_ROOT, "plates"), table="event_plates",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads", camera_cols=["camera_id"],
    ),
    Category(
        key="parking_lot_event", weight_field="w_parking_lot_event",
        directory=os.path.join(UPLOADS_ROOT, "parking"), table="parking_lot_event",
        order_col="timestamp", file_cols=["face_image_full", "plate_image_full"],
        file_kind="uploads",
        # Một sự kiện bãi xe thuộc về HAI camera. Xoá nó theo hạn của riêng
        # camera mặt là xoá luôn bằng chứng của camera biển số còn trong hạn.
        camera_cols=["face_camera_id", "plate_camera_id"],
    ),
    Category(
        key="restricted_area", weight_field="w_restricted_area",
        directory=os.path.join(UPLOADS_ROOT, "restricted"), table="restricted_areas",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads", camera_cols=["camera_id"],
    ),
    Category(
        key="event_mask", weight_field="w_event_mask",
        directory=os.path.join(UPLOADS_ROOT, "masks"), table="event_mask",
        order_col="timestamp", file_cols=["image_full", "image_crop"],
        file_kind="uploads", camera_cols=["camera_id"],
    ),
    Category(
        # Sự kiện chuyển động: ảnh do engine C++ ghi nên đường dẫn tương đối
        # với gstreamer_c/ (giống recording), không phải /uploads.
        key="motion_event", weight_field="w_motion_event",
        directory=MOTION_SNAPSHOTS_DIR, table="motion_events",
        order_col="start_at", file_cols=["image_path"],
        file_kind="recording",
        # Sự kiện vừa xảy ra có thể chưa kịp ghi xong ảnh (engine chèn hàng lúc
        # sự kiện KẾT THÚC, ảnh chụp lúc nó BẮT ĐẦU). Chừa một khoảng an toàn.
        where="start_at < now() - interval '3 minutes'",
        time_kind="tstz", camera_cols=["camera_id"],
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
    # Bao nhiêu hàng trong deleted_rows là do HẠN NGÀY (chứ không do đĩa đầy).
    # Tách ra để log nói được vì sao dữ liệu biến mất — hai luật xoá cùng một
    # bảng, không tách thì không lần ra được nguyên nhân.
    retention_rows: int = 0


class StorageCleanupService:
    async def _load_policy(self, session: AsyncSession):
        # Cột trọng số lấy thẳng từ CATEGORIES: thêm một loại chỉ còn phải sửa
        # đúng một chỗ (danh sách ở trên) thay vì nhớ sửa cả câu SELECT này.
        weights = ", ".join(c.weight_field for c in CATEGORIES)
        row = (await session.execute(text(
            f"SELECT enabled, min_free_gb, target_free_gb, {weights} "
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

    async def _delete_batch(
        self,
        session: AsyncSession,
        cat: Category,
        n: int,
        *,
        cte: str = "",
        join: str = "",
        extra_where: str = "",
        params: dict = None,
    ):
        """Xoá n hàng CŨ NHẤT của một loại (kèm file). Trả (bytes, số hàng).

        cte/join/extra_where để lượt HẠN NGÀY gắn thêm bảng hạn của từng camera
        vào cùng một câu lệnh; lượt đĩa-đầy gọi không kèm gì và ra đúng câu lệnh
        như trước.
        """
        conds = [c for c in (cat.where, extra_where) if c]
        where = ("WHERE " + " AND ".join(f"({c})" for c in conds)) if conds else ""
        cols = ", ".join(f"t.{c}" for c in cat.file_cols)
        args = {"n": n, **(params or {})}
        rows = (await session.execute(text(
            f"{cte} SELECT CAST(t.id AS text) AS id, {cols} FROM {cat.table} t {join} "
            f"{where} ORDER BY t.{cat.order_col} ASC LIMIT :n"
        ), args)).mappings().all()
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

    # ---------- Luật 1: hạn lưu theo NGÀY, riêng từng camera ----------

    async def _load_retentions(self, session: AsyncSession) -> List[dict]:
        """{id, retention_days} của các camera CÓ đặt hạn (days > 0).

        Chịu được máy chưa từng chạy engine C++: bảng `cameras` do engine tạo
        (sql/001_init_cameras.sql) chứ không phải create_all của Python, nên
        thiếu bảng/thiếu cột là chuyện bình thường, không phải lỗi.
        """
        try:
            rows = (await session.execute(text(
                "SELECT CAST(id AS text) AS id, retention_days FROM cameras "
                "WHERE retention_days > 0"
            ))).mappings().all()
            return [dict(r) for r in rows]
        except Exception:
            await session.rollback()
            return []

    def _retention_clauses(self, cat: Category):
        """(join, where) cho hạn ngày. `r(cam, days)` do run_retention dựng.

        Một hàng HẾT HẠN khi: mọi cột camera có giá trị đều tra được trong r,
        có ít nhất một cột tra được, và hàng đã già hơn hạn LỚN NHẤT trong số
        đó. LEFT JOIN + đòi `days IS NOT NULL` là cách nói "camera nào không
        đặt hạn thì giữ vô thời hạn" — dùng INNER JOIN sẽ im lặng bỏ sót,
        còn bỏ điều kiện đó đi thì sự kiện bãi xe bị camera này xoá mất phần
        của camera kia.
        """
        joins, guards, days = [], [], []
        for i, col in enumerate(cat.camera_cols):
            key = f"nullif(CAST(t.{col} AS text), '')"
            joins.append(f"LEFT JOIN r r{i} ON {key} = r{i}.cam")
            guards.append(f"({key} IS NULL OR r{i}.days IS NOT NULL)")
            days.append(f"r{i}.days")
        # greatest() bỏ qua NULL, nên cột camera trống không kéo hạn về 0.
        max_days = f"greatest({', '.join(f'coalesce({d}, 0)' for d in days)})"
        cutoff = f"now() - make_interval(days => {max_days})"
        if cat.time_kind == "epoch":
            cutoff = f"extract(epoch from {cutoff})"
        guards.append(f"coalesce({', '.join(days)}) IS NOT NULL")
        guards.append(f"t.{cat.order_col} < {cutoff}")
        return " ".join(joins), " AND ".join(guards)

    async def run_retention(self, session: AsyncSession, stats: _RunStats = None) -> _RunStats:
        """Xoá mọi thứ đã quá hạn lưu của camera sở hữu nó.

        KHÔNG phụ thuộc storage_policy.enabled và không nhìn chỗ trống của ổ:
        hạn lưu là lời hứa "chỉ giữ N ngày", còn công tắc kia chỉ nói về chuyện
        chữa cháy khi đĩa sắp đầy. Cũng KHÔNG phụ thuộc camera có đang ghi hay
        đang kết nối không — lượt này chỉ đọc DB.
        """
        stats = stats or _RunStats()
        cams = await self._load_retentions(session)
        if not cams:
            return stats

        params = {}
        values = []
        for i, cam in enumerate(cams):
            params[f"c{i}"] = cam["id"]
            params[f"d{i}"] = int(cam["retention_days"])
            values.append(f"(CAST(:c{i} AS text), CAST(:d{i} AS int))")
        cte = f"WITH r(cam, days) AS (VALUES {', '.join(values)})"

        for cat in CATEGORIES:
            if not cat.camera_cols:
                continue
            join, where = self._retention_clauses(cat)
            guard = 0
            while guard < 500:  # trần 100k hàng/loại/chu kỳ, đủ để không treo
                guard += 1
                try:
                    freed, ndel = await self._delete_batch(
                        session, cat, _BATCH,
                        cte=cte, join=join, extra_where=where, params=params,
                    )
                except Exception as exc:
                    await session.rollback()
                    print(f"[storage] han luu {cat.key} loi: {exc}")
                    traceback.print_exc()
                    break
                if ndel == 0:
                    break
                stats.freed_bytes += freed
                stats.deleted_rows += ndel
                stats.retention_rows += ndel
                stats.per_category[cat.key] = stats.per_category.get(cat.key, 0) + freed
        return stats

    # ---------- Luật 2: giữ tối thiểu N GB trống ----------

    async def run_once(self, session: AsyncSession) -> _RunStats:
        # Hạn ngày chạy TRƯỚC và luôn chạy: dọn xong phần quá hạn rồi mới đo
        # chỗ trống, nhờ vậy nhiều hôm không phải đụng tới dữ liệu còn trong hạn.
        stats = await self.run_retention(session)
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
