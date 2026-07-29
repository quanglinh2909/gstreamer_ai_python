# -*- coding: utf-8 -*-
"""Truy xuất khung phát hiện đã lưu: vẽ lại khi XEM LẠI, và TÌM theo vùng vẽ.

Hai đầu vào khác nhau nhưng cùng một bảng và cùng một chỉ mục
(camera_id, t_start) — xem app/models/detection_slice.py.
"""

import struct
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.detection_slice import DetectionSlice

router = APIRouter()
prefix = "/detections"
tags = ["Detections"]

GRID = 16
# Trần số lát trả về một lần. Một camera ~26k lát/ngày, nên phải chặn để một
# yêu cầu khoảng thời gian rộng không kéo cả ngày về trình duyệt.
MAX_SLICES = 4000


class SampleDTO(BaseModel):
    t: int                      # epoch ms
    b: List[float]              # [x1, y1, x2, y2] chuẩn hoá [0,1]
    s: float                    # score
    k: Optional[List[float]] = None   # pose phẳng (x, y, score) × K


class TrackDTO(BaseModel):
    tid: Optional[int]
    ai_type: Optional[str]
    class_id: Optional[int]
    t_start: int
    t_end: int
    best_score: Optional[float]
    samples: List[SampleDTO]


def _decode_path(blob: bytes, t0: int):
    """Giải mã quỹ đạo nhị phân -> [(t, x1, y1, x2, y2, score)]."""
    if not blob or len(blob) < 3:
        return []
    ver, n = struct.unpack_from("<BH", blob, 0)
    if ver != 1:
        return []
    out = []
    off = 3
    t = t0
    for _ in range(n):
        if off + 11 > len(blob):
            break
        dt, x1, y1, x2, y2, s = struct.unpack_from("<HHHHHB", blob, off)
        off += 11
        t += dt
        out.append((t, x1 / 65535, y1 / 65535, x2 / 65535, y2 / 65535, s / 255))
    return out


def _decode_kps(blob: Optional[bytes]):
    """Giải mã pose -> [[x, y, score] × K] cho từng mẫu."""
    if not blob or len(blob) < 4:
        return []
    ver, k, n = struct.unpack_from("<BBH", blob, 0)
    if ver != 1 or k == 0:
        return []
    out = []
    off = 4
    for _ in range(n):
        flat = []
        for _ in range(k):
            if off + 5 > len(blob):
                return out
            x, y, s = struct.unpack_from("<HHB", blob, off)
            off += 5
            flat.extend([round(x / 65535, 4), round(y / 65535, 4), round(s / 255, 2)])
        out.append(flat)
    return out


def _cells_of_rect(x1: float, y1: float, x2: float, y2: float) -> bytearray:
    """Lưới 16×16 mà hình chữ nhật này phủ — cùng quy ước với bộ ghi."""
    mask = bytearray(GRID * GRID // 8)
    gx0 = max(0, min(GRID - 1, int(x1 * GRID)))
    gx1 = max(0, min(GRID - 1, int(x2 * GRID)))
    gy0 = max(0, min(GRID - 1, int(y1 * GRID)))
    gy1 = max(0, min(GRID - 1, int(y2 * GRID)))
    for gy in range(gy0, gy1 + 1):
        for gx in range(gx0, gx1 + 1):
            bit = gy * GRID + gx
            mask[bit >> 3] |= 1 << (bit & 7)
    return mask


def _masks_overlap(a: Optional[bytes], b: bytearray) -> bool:
    if not a:
        return True          # không có lưới (dữ liệu cũ) -> đành tin bbox
    return any(x & y for x, y in zip(a, b))


@router.get("/tracks", response_model=List[TrackDTO])
async def list_tracks(
    camera_id: str = Query(...),
    from_ms: int = Query(..., description="epoch ms, bao gồm"),
    to_ms: int = Query(..., description="epoch ms, bao gồm"),
    db: AsyncSession = Depends(get_db),
):
    """Quỹ đạo để VẼ LẠI khi xem bản ghi, trong khoảng thời gian yêu cầu.

    Trả từng mẫu kèm mốc thời gian tuyệt đối để client chỉ việc tra theo con
    trỏ phát; các lát của cùng một `tid` được gộp lại thành một track."""
    if to_ms < from_ms:
        raise HTTPException(400, "to_ms phải >= from_ms")
    rows = (await db.execute(
        select(DetectionSlice)
        .where(
            DetectionSlice.camera_id == camera_id,
            DetectionSlice.t_end >= from_ms,
            DetectionSlice.t_start <= to_ms,
        )
        .order_by(DetectionSlice.t_start)
        .limit(MAX_SLICES)
    )).scalars().all()

    # Gộp các lát liền nhau của cùng một track thành MỘT mục.
    merged = {}
    for r in rows:
        key = (r.tid, r.job_id)
        pts = _decode_path(r.path, r.t_start)
        kps = _decode_kps(r.kps)
        samples = []
        for i, (t, x1, y1, x2, y2, s) in enumerate(pts):
            if t < from_ms or t > to_ms:
                continue
            samples.append(SampleDTO(
                t=t, b=[round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)],
                s=round(s, 3), k=kps[i] if i < len(kps) else None,
            ))
        if not samples:
            continue
        hit = merged.get(key)
        if hit is None:
            merged[key] = TrackDTO(
                tid=r.tid, ai_type=r.ai_type, class_id=r.class_id,
                t_start=samples[0].t, t_end=samples[-1].t,
                best_score=r.best_score, samples=samples,
            )
        else:
            hit.samples.extend(samples)
            hit.t_end = samples[-1].t
            if (r.best_score or 0) > (hit.best_score or 0):
                hit.best_score = r.best_score
    return list(merged.values())


class RegionHitDTO(BaseModel):
    tid: Optional[int]
    ai_type: Optional[str]
    class_id: Optional[int]
    t_start: int
    t_end: int
    best_score: Optional[float]
    # bbox hợp của phần track nằm trong khoảng — để hiện ảnh xem trước.
    bbox: List[float]


@router.get("/region-search", response_model=List[RegionHitDTO])
async def region_search(
    camera_id: str = Query(...),
    from_ms: int = Query(...),
    to_ms: int = Query(...),
    x1: float = Query(..., ge=0, le=1),
    y1: float = Query(..., ge=0, le=1),
    x2: float = Query(..., ge=0, le=1),
    y2: float = Query(..., ge=0, le=1),
    ai_type: Optional[str] = Query(None),
    gap_ms: int = Query(5000, description="Cách nhau quá lâu thì tách sự kiện"),
    db: AsyncSession = Depends(get_db),
):
    """Tìm những gì ĐÃ ĐI QUA vùng vẽ trên hình, trong khoảng thời gian.

    Ba tầng lọc, từ rẻ đến đắt:
      1. chỉ mục (camera_id, t_start) chặn theo camera + thời gian;
      2. giao nhau hình chữ nhật bằng 4 phép so sánh trên bbox của lát;
      3. AND lưới 16×16 (32 byte) — bbox là bao lồi nên người đi chéo màn hình
         khớp cả những vùng họ chưa từng bước vào; tầng này loại chúng mà
         KHÔNG phải giải mã quỹ đạo.
    Cuối cùng gộp các lát cùng `tid` cách nhau < gap_ms thành một sự kiện."""
    if to_ms < from_ms:
        raise HTTPException(400, "to_ms phải >= from_ms")
    qx1, qx2 = min(x1, x2), max(x1, x2)
    qy1, qy2 = min(y1, y2), max(y1, y2)

    stmt = select(DetectionSlice).where(
        DetectionSlice.camera_id == camera_id,
        DetectionSlice.t_end >= from_ms,
        DetectionSlice.t_start <= to_ms,
        # Giao nhau hình chữ nhật (không phải "nằm trong"): chỉ cần chạm.
        DetectionSlice.bx1 <= qx2,
        DetectionSlice.bx2 >= qx1,
        DetectionSlice.by1 <= qy2,
        DetectionSlice.by2 >= qy1,
    )
    if ai_type:
        stmt = stmt.where(DetectionSlice.ai_type == ai_type)
    rows = (await db.execute(
        stmt.order_by(DetectionSlice.t_start).limit(MAX_SLICES)
    )).scalars().all()

    qmask = _cells_of_rect(qx1, qy1, qx2, qy2)
    hits: List[RegionHitDTO] = []
    last_by_key = {}
    for r in rows:
        if not _masks_overlap(r.cells, qmask):
            continue
        key = (r.tid, r.job_id)
        prev = last_by_key.get(key)
        if prev is not None and r.t_start - prev.t_end <= gap_ms:
            prev.t_end = max(prev.t_end, r.t_end)
            prev.bbox = [
                min(prev.bbox[0], r.bx1), min(prev.bbox[1], r.by1),
                max(prev.bbox[2], r.bx2), max(prev.bbox[3], r.by2),
            ]
            if (r.best_score or 0) > (prev.best_score or 0):
                prev.best_score = r.best_score
            continue
        hit = RegionHitDTO(
            tid=r.tid, ai_type=r.ai_type, class_id=r.class_id,
            t_start=r.t_start, t_end=r.t_end, best_score=r.best_score,
            bbox=[r.bx1, r.by1, r.bx2, r.by2],
        )
        hits.append(hit)
        last_by_key[key] = hit
    # MỚI NHẤT TRƯỚC — cùng thứ tự với bảng sự kiện, người xem quét từ trên
    # xuống là đi ngược dòng thời gian. Vòng gộp ở trên VẪN cần quét tăng dần
    # (nó dựa vào việc lát sau luôn tới sau lát trước), nên chỉ đảo ở đây.
    hits.sort(key=lambda h: h.t_start, reverse=True)
    return hits
