# -*- coding: utf-8 -*-
"""Truy xuất khung phát hiện đã lưu: vẽ lại khi XEM LẠI, và TÌM theo vùng vẽ.

Hai đầu vào đầu dùng chung bảng `detection_slice` và chỉ mục
(camera_id, t_start) — xem app/models/detection_slice.py.

Tìm theo vùng còn quét THÊM bảng `motion_events` (engine C++ ghi) và trả kết
quả chuyển động trong CÙNG một danh sách, gắn `ai_type="motion"`. Hai nguồn này
không gộp được ở tầng SQL: một bên là track có bbox trên lưới 16×16 cố định,
bên kia là tập ô trên lưới riêng của từng camera.
"""

import struct
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, text
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
# Sự kiện chuyển động thưa hơn lát AI rất nhiều (một sự kiện kéo dài vài giây
# thay vì một lát mỗi 2 giây cho MỖI vật thể), nên trần thấp hơn là đủ.
MAX_MOTION_EVENTS = 2000
# `ai_type` quy ước cho chuyển động. KHÔNG phải một loại AI thật — nó không đi
# qua model nào — nhưng dùng chung ô này để kết quả trộn được vào một danh sách.
MOTION_TYPE = "motion"


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
    # ---- Chỉ có ở kết quả CHUYỂN ĐỘNG (lát AI để trống) ----
    # Id hàng motion_events: giao diện dùng nó để lấy đúng ảnh engine đã chụp
    # lúc sự kiện bắt đầu, thay vì trích lại thumbnail từ bản ghi.
    event_id: Optional[str] = None
    # Ô đã động + cỡ lưới CỦA CHÍNH sự kiện, để vẽ lớp phủ giống bảng sự kiện.
    cells: Optional[str] = None
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None


@router.get("/region-search", response_model=List[RegionHitDTO])
async def region_search(
    camera_id: str = Query(...),
    from_ms: int = Query(...),
    to_ms: int = Query(...),
    x1: float = Query(..., ge=0, le=1),
    y1: float = Query(..., ge=0, le=1),
    x2: float = Query(..., ge=0, le=1),
    y2: float = Query(..., ge=0, le=1),
    ai_type: Optional[str] = Query(
        None, description='Lọc một loại; "motion" = chỉ sự kiện chuyển động'),
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
    Cuối cùng gộp các lát cùng `tid` cách nhau < gap_ms thành một sự kiện.

    CHUYỂN ĐỘNG cũng tìm được ở đây (`ai_type="motion"`) nhưng đi đường riêng:
    nó nằm ở bảng `motion_events` của engine C++, không có track/bbox mà chỉ có
    tập ô đã động — xem `_motion_hits`."""
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

    # CHUYỂN ĐỘNG: bảng khác, đơn vị khác, nên quét riêng rồi trộn vào.
    if not ai_type or ai_type == MOTION_TYPE:
        hits.extend(await _motion_hits(db, camera_id, from_ms, to_ms,
                                       qx1, qy1, qx2, qy2))

    # MỚI NHẤT TRƯỚC — cùng thứ tự với bảng sự kiện, người xem quét từ trên
    # xuống là đi ngược dòng thời gian. Vòng gộp ở trên VẪN cần quét tăng dần
    # (nó dựa vào việc lát sau luôn tới sau lát trước), nên chỉ đảo ở đây.
    hits.sort(key=lambda h: h.t_start, reverse=True)
    return hits


async def _motion_hits(db: AsyncSession, camera_id: str, from_ms: int, to_ms: int,
                       qx1: float, qy1: float, qx2: float,
                       qy2: float) -> List[RegionHitDTO]:
    """Sự kiện CHUYỂN ĐỘNG có ô nào chạm vùng đã khoanh.

    Vì sao không dùng chung đường với AI:
      * `motion_events` do engine C++ ghi, không có `tid` (không bám vật thể),
        không có bbox, không có class — chỉ một tập Ô ĐÃ ĐỘNG;
      * lưới của nó là lưới của camera (thường 32×32), khác lưới 16×16 cố định
        mà lát AI dùng. May là mỗi hàng tự mang `grid_x/grid_y` của chính nó,
        nên đổi độ phân giải lưới giữa chừng không làm dữ liệu cũ lệch.

    Ô (r, c) phủ đúng [c/gx, (c+1)/gx] × [r/gy, (r+1)/gy] trong khung hình.
    KHÔNG phải trừ viền letterbox: MotionDetector trải lưới trên PHẦN ẢNH THẬT
    chứ không trên cả khung có đệm đen (xem MotionDetector::submit).

    Đọc bằng SQL thô: bảng này thuộc quyền engine C++, khai báo lại thành model
    SQLAlchemy là mời create_all() đụng vào một bảng nó không sở hữu.
    """
    rows = (await db.execute(
        text(
            "SELECT id, cells, grid_x, grid_y, image_path,"
            "       (EXTRACT(EPOCH FROM start_at) * 1000)::bigint AS t_start,"
            "       (EXTRACT(EPOCH FROM COALESCE(end_at, start_at)) * 1000)::bigint AS t_end"
            "  FROM motion_events"
            " WHERE camera_id = CAST(:cam AS uuid)"
            "   AND start_at <= to_timestamp(:to_ms / 1000.0)"
            "   AND COALESCE(end_at, start_at) >= to_timestamp(:from_ms / 1000.0)"
            " ORDER BY start_at DESC LIMIT :lim"
        ),
        {"cam": camera_id, "from_ms": from_ms, "to_ms": to_ms,
         "lim": MAX_MOTION_EVENTS},
    )).mappings().all()

    out: List[RegionHitDTO] = []
    for row in rows:
        gx = row["grid_x"] or 0
        gy = row["grid_y"] or 0
        if gx <= 0 or gy <= 0 or not row["cells"]:
            continue
        bx1 = by1 = 1.0
        bx2 = by2 = 0.0
        touched = False
        for token in row["cells"].split(","):
            r, _, c = token.partition(":")
            try:
                ri, ci = int(r), int(c)
            except ValueError:
                continue
            cx1, cx2 = ci / gx, (ci + 1) / gx
            cy1, cy2 = ri / gy, (ri + 1) / gy
            # Chồng lấn THỰC SỰ (< > chứ không <= >=): vẽ đúng trên đường biên
            # ô thì không nên tính là đã đi qua ô bên cạnh.
            if cx1 >= qx2 or cx2 <= qx1 or cy1 >= qy2 or cy2 <= qy1:
                continue
            touched = True
            bx1, by1 = min(bx1, cx1), min(by1, cy1)
            bx2, by2 = max(bx2, cx2), max(by2, cy2)
        if not touched:
            continue
        out.append(RegionHitDTO(
            tid=None, ai_type=MOTION_TYPE, class_id=None,
            t_start=int(row["t_start"]), t_end=int(row["t_end"]),
            # Không có độ tin cậy thật — để trống thay vì bịa một con số.
            best_score=None,
            bbox=[bx1, by1, bx2, by2],
            event_id=str(row["id"]),
            cells=row["cells"],
            grid_x=gx,
            grid_y=gy,
        ))
    return out
