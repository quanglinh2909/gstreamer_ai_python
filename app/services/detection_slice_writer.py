"""Gom khung phát hiện theo TRACK × LÁT THỜI GIAN rồi ghi xuống DB theo mẻ.

Chỉ chạy cho job bật `save_detections` (mặc định TẮT). Mục đích: xem lại bản
ghi thì vẽ được box/pose, và vẽ một vùng trên hình để tìm sự kiện đã đi qua đó.

Nguyên tắc giữ cho rẻ:
  - Không ghi mỗi khung một dòng (~1 triệu dòng/ngày/job) mà gom thành lát
    SLICE_MS; mỗi (track, lát) là MỘT dòng kèm quỹ đạo nén nhị phân.
  - Chèn theo MẺ, và luôn chạy ngoài vòng nhận AI (fire-and-forget) để một cú
    ghi DB chậm không làm nghẽn luồng xử lý ảnh.
"""

from __future__ import annotations

import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

from app.models.detection_slice import DetectionSlice

SLICE_MS = 10_000
# Track im lặng quá lâu thì chốt lát đang dở (người đi khỏi khung).
TRACK_IDLE_MS = 3_000
# Trần số mẫu một lát, chặn trường hợp fps cấu hình cao bất thường.
MAX_SAMPLES = 1_000
GRID = 16


def _u16(v: float) -> int:
    return max(0, min(65535, int(round(v * 65535))))


def _u8(v: float) -> int:
    return max(0, min(255, int(round(v * 255))))


class _Acc:
    """Một track đang được gom trong một lát."""

    __slots__ = ("t0", "t_last", "samples", "kps", "kps_k", "class_id",
                 "ai_type", "best", "bx1", "by1", "bx2", "by2", "cells")

    def __init__(self, t: int, ai_type, class_id):
        self.t0 = t
        self.t_last = t
        self.samples: List[Tuple[int, float, float, float, float, float]] = []
        self.kps: List[List[float]] = []
        self.kps_k = 0
        self.class_id = class_id
        self.ai_type = ai_type
        self.best = 0.0
        self.bx1, self.by1, self.bx2, self.by2 = 1.0, 1.0, 0.0, 0.0
        self.cells = bytearray(GRID * GRID // 8)

    def add(self, t: int, b: dict) -> None:
        if len(self.samples) >= MAX_SAMPLES:
            return
        x1, y1 = float(b["x1"]), float(b["y1"])
        x2, y2 = float(b["x2"]), float(b["y2"])
        s = float(b.get("score") or 0.0)
        self.samples.append((t, x1, y1, x2, y2, s))
        self.t_last = t
        self.best = max(self.best, s)
        self.bx1 = min(self.bx1, x1); self.by1 = min(self.by1, y1)
        self.bx2 = max(self.bx2, x2); self.by2 = max(self.by2, y2)
        # Đánh dấu các ô lưới mà box này phủ.
        gx0 = max(0, min(GRID - 1, int(x1 * GRID)))
        gx1 = max(0, min(GRID - 1, int(x2 * GRID)))
        gy0 = max(0, min(GRID - 1, int(y1 * GRID)))
        gy1 = max(0, min(GRID - 1, int(y2 * GRID)))
        for gy in range(gy0, gy1 + 1):
            row = gy * GRID
            for gx in range(gx0, gx1 + 1):
                bit = row + gx
                self.cells[bit >> 3] |= 1 << (bit & 7)
        kp = b.get("kps")
        if kp:
            k = len(kp) // 3
            if self.kps_k == 0:
                self.kps_k = k
            if k == self.kps_k:
                self.kps.append(kp)

    def encode_path(self) -> bytes:
        out = bytearray(struct.pack("<BH", 1, len(self.samples)))
        prev = self.t0
        for (t, x1, y1, x2, y2, s) in self.samples:
            dt = max(0, min(65535, t - prev)); prev = t
            out += struct.pack("<HHHHHB", dt, _u16(x1), _u16(y1), _u16(x2), _u16(y2), _u8(s))
        return bytes(out)

    def encode_kps(self) -> Optional[bytes]:
        # Chỉ lưu pose khi MỌI mẫu đều có, để chỉ số mẫu của kps khớp path.
        if not self.kps or self.kps_k == 0 or len(self.kps) != len(self.samples):
            return None
        out = bytearray(struct.pack("<BBH", 1, self.kps_k, len(self.kps)))
        for kp in self.kps:
            for i in range(self.kps_k):
                out += struct.pack("<HHB", _u16(kp[i * 3]), _u16(kp[i * 3 + 1]),
                                   _u8(kp[i * 3 + 2]))
        return bytes(out)


class DetectionSliceWriter:
    def __init__(self) -> None:
        # (camera_id, job_id, tid, chỉ-số-lát) -> _Acc
        self._acc: Dict[Tuple[str, str, int, int], _Acc] = {}
        self._pending: List[dict] = []
        self._last_flush = 0.0

    def add_frame(self, camera_id: str, job_id: str, ai_type, boxes: List[dict],
                  ts_ms: Optional[int] = None) -> None:
        """Nạp một khung. Chỉ gọi khi job BẬT save_detections."""
        t = int(ts_ms if ts_ms is not None else time.time() * 1000)
        for b in boxes:
            tid = b.get("tid")
            if tid is None:
                # Không có id theo dõi thì không gom thành quỹ đạo được; bỏ qua
                # (mọi tracker của hệ đều phát id, đây chỉ là lưới an toàn).
                continue
            key = (camera_id, job_id, int(tid), t // SLICE_MS)
            acc = self._acc.get(key)
            if acc is None:
                acc = _Acc(t, ai_type, b.get("class_id"))
                self._acc[key] = acc
            acc.add(t, b)

    def collect(self, now_ms: Optional[int] = None) -> List[dict]:
        """Chốt các lát đã qua / track đã im, trả về danh sách dòng để chèn."""
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        cur_slice = now // SLICE_MS
        done = []
        for key, acc in list(self._acc.items()):
            _, _, _, sl = key
            if sl < cur_slice or now - acc.t_last > TRACK_IDLE_MS:
                self._acc.pop(key, None)
                if acc.samples:
                    done.append(self._row(key, acc))
        return done

    @staticmethod
    def _row(key, acc: _Acc) -> dict:
        camera_id, job_id, tid, _ = key
        return dict(
            camera_id=camera_id, job_id=job_id, ai_type=acc.ai_type,
            class_id=acc.class_id, tid=tid,
            t_start=acc.t0, t_end=acc.t_last,
            bx1=acc.bx1, by1=acc.by1, bx2=acc.bx2, by2=acc.by2,
            cells=bytes(acc.cells), best_score=acc.best, n=len(acc.samples),
            path=acc.encode_path(), kps=acc.encode_kps(),
        )

    async def flush(self, session_factory, now_ms: Optional[int] = None) -> int:
        rows = self.collect(now_ms)
        if not rows:
            return 0
        try:
            async with session_factory() as db:
                db.add_all([DetectionSlice(**r) for r in rows])
                await db.commit()
            return len(rows)
        except Exception as exc:
            # Mất vài lát khung phát hiện KHÔNG được phép làm chết luồng AI.
            print(f"detection slice flush failed: {exc}", file=sys.stderr)
            return 0


detection_slice_writer = DetectionSliceWriter()
