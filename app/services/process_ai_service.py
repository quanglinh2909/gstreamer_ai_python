import asyncio
import json
import socket
import struct
import sys
import threading
import time
from typing import Optional

import numpy as np
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.repositories.ai_config_repository import AIRepository
from app.utils.process_ai_hepper import ProcessAiHepper
from app.ws.live_detection_ws import live_detection_broadcaster
from app.services.detection_slice_writer import detection_slice_writer

# Trần số khung phát hiện gửi kèm một khung hình. Cảnh đông người mà gửi hết
# thì gói phình ra vô ích — nhìn trên màn hình cũng chỉ là một rừng khung.
MAX_LIVE_BOXES = 50

# VÙNG (polygon) là dữ liệu TĨNH của cấu hình, gửi kèm mỗi khung hình thì gần
# như gấp đôi gói mà chẳng thêm thông tin gì. Gửi lại mỗi ngần này giây thôi;
# client giữ vùng đã nhận và chỉ cập nhật khi có khoá `zones` mới. Người mới mở
# xem chờ tối đa ngần này là thấy vùng.
ZONE_RESEND_SEC = 1.0


class ProcessAiService:
    def __init__(self, url_socket="/tmp/ai_engine.sock", reconnect_delay=2.0):
        self.url_socket = url_socket
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.sock = None
        self.process_ai = {}
        self._session_factory = None
        # API thread schedules cache invalidations here; recv loop drains
        # them before each message. (camera_id, job_id) — job_id=None means
        # "every cached job for that camera" (used on camera delete).
        self._invalidate_lock = threading.Lock()
        self._pending_invalidations: set = set()
        # Debug stream subscribers, ref-counted per (cam, job). Hot path
        # checks the *dict* directly: when no one is watching it costs a
        # single O(1) `key in dict` check (atomic in CPython) — no copy,
        # no JPEG decode, no locking. Only when at least one MJPEG client
        # is connected do we stash the latest meta+jpeg+polygons.
        # Lần cuối gửi VÙNG cho mỗi (cam, job) — vùng là dữ liệu tĩnh nên chỉ
        # gửi lại thưa (ZONE_RESEND_SEC) thay vì kèm mỗi khung hình.
        self._zone_sent_at: dict = {}        # (cam, job) -> epoch giây
        # Nhịp ghi detection_slice xuống DB (chạy nền, không chặn vòng nhận).
        self._last_slice_flush = 0.0
        self._debug_lock = threading.Lock()
        self._debug_subscribers: dict = {}   # (cam, job) -> refcount
        self._debug_latest: dict = {}        # (cam, job) -> {meta, full_jpeg, seq, polygons, primary_conf}
        # Ảnh JPEG đầy MỚI NHẤT theo camera. Engine giới hạn tần suất encode JPEG
        # (encode phần mềm 1080p tốn CPU) nên có khung metadata KHÔNG kèm ảnh.
        # Sự kiện rơi đúng khung đó vẫn cần ảnh -> lấy ảnh mới nhất (trễ tối đa
        # ~vài trăm ms, thừa đủ cho ảnh sự kiện). camera_id -> bytes.
        self._last_full_jpeg: dict = {}
        # Dedup the "AI config not found" warning: C++ engine workers that
        # never had a Python-side AIConfig (typically orphan jobs left
        # behind by the old duplicate-on-save bug) would otherwise spam
        # this line every frame. Log once per (cam, job) until that pair
        # is either resolved or this process restarts.
        self._unknown_job_logged: set = set()

    def invalidate(self, camera_id: str, job_id: Optional[str] = None) -> None:
        """Drop the cached per-(camera, job) state so the next inbound
        message reloads tracker, polygons, thresholds, etc. from DB.

        Safe to call from any thread (API handlers run on the FastAPI loop;
        the recv loop runs on its own thread). Pass job_id=None to flush
        every job belonging to the camera."""
        if camera_id is None:
            return
        key = (str(camera_id), str(job_id) if job_id is not None else None)
        with self._invalidate_lock:
            self._pending_invalidations.add(key)

    # ─── Debug MJPEG hooks ────────────────────────────────────────────
    def subscribe_debug(self, camera_id: str, job_id: str) -> None:
        key = (str(camera_id), str(job_id))
        with self._debug_lock:
            self._debug_subscribers[key] = self._debug_subscribers.get(key, 0) + 1

    def unsubscribe_debug(self, camera_id: str, job_id: str) -> None:
        key = (str(camera_id), str(job_id))
        with self._debug_lock:
            cur = self._debug_subscribers.get(key, 0)
            if cur <= 1:
                self._debug_subscribers.pop(key, None)
                # Free the cached frame as soon as nobody is watching, so
                # idle debug doesn't keep the last JPEG around forever.
                self._debug_latest.pop(key, None)
            else:
                self._debug_subscribers[key] = cur - 1

    def get_latest_debug_frame(self, camera_id: str, job_id: str) -> Optional[dict]:
        """Snapshot of the most recent cached frame for this (cam, job),
        or None when no frame has been stashed yet."""
        with self._debug_lock:
            hit = self._debug_latest.get((str(camera_id), str(job_id)))
            return dict(hit) if hit else None

    def _stash_debug_frame(self, camera_id, job_id, meta, full_jpeg, polygons,
                           primary_conf=0.3, overlap_threshold=None,
                           class_meta=None) -> None:
        """Hot path. Returns immediately when nobody is debugging this
        (cam, job). Stores references only — no copies — because the
        recv loop is done mutating `meta` by the time this is called and
        `full_jpeg` is immutable bytes from the C++ side."""
        # The C++ AI engine only encodes a JPEG when the frame has at
        # least one detection (`if (!res.detections.empty()) encodeImages`).
        # Empty-payload messages still come through for stats / FPS, but
        # there's nothing to render — and trying to imdecode b"" later
        # would assert in OpenCV. Drop them at the door.
        if not full_jpeg:
            return
        key = (str(camera_id), str(job_id))
        # Lock-free fast exit. Dict membership is atomic in CPython, so
        # a stale True/False here would at worst trigger one extra stash
        # cycle — never corrupts state.
        if key not in self._debug_subscribers:
            return
        with self._debug_lock:
            if key not in self._debug_subscribers:
                return
            self._debug_latest[key] = {
                "meta": meta,
                "full_jpeg": full_jpeg,
                "seq": meta.get("seq", 0),
                "polygons": polygons,
                "primary_conf": primary_conf,
                # So the overlay's green/red in-zone verdict is drawn with
                # the same threshold the recv loop judged the frame by.
                "overlap_threshold": overlap_threshold,
                # Optional per-class name/color the overlay uses to relabel
                # and recolor boxes (None for services that don't declare it).
                "class_meta": class_meta,
            }

    @staticmethod
    def _build_boxes(meta, class_meta, min_score=0.0):
        """Khung phát hiện của một khung hình, CHUẨN HOÁ [0,1].

        Tách riêng vì có HAI người dùng: đẩy websocket cho người xem trực tiếp,
        và bộ ghi detection_slice. Dựng một lần rồi dùng chung — trước đây nằm
        trong _publish_live_boxes nên bật ghi mà không ai xem thì không có gì.

        Trả (boxes, width, height); width=0 nghĩa là khung không dùng được."""
        # Engine C++ gửi `origWidth`/`origHeight` (xem ResultPublisher::
        # buildJson) — KHÔNG phải `width`/`height`. Vẫn đọc cả hai tên
        # phòng khi định dạng đổi.
        width = float(meta.get("origWidth") or meta.get("width") or 0)
        height = float(meta.get("origHeight") or meta.get("height") or 0)
        if width <= 0 or height <= 0:
            return [], 0.0, 0.0

        def norm(value, size):
            # Kẹp về [0,1]: bbox có thể lòi ra ngoài mép khung.
            return round(min(1.0, max(0.0, float(value) / size)), 4)

        boxes = []
        for det in (meta.get("detections") or []):
            if len(boxes) >= MAX_LIVE_BOXES:
                break
            score = float(det.get("score", 0.0))
            if score < min_score:
                continue
            class_id = det.get("classId")
            entry = (class_meta or {}).get(class_id) or {}
            box = {
                "x1": norm(det.get("x1", 0), width),
                "y1": norm(det.get("y1", 0), height),
                "x2": norm(det.get("x2", 0), width),
                "y2": norm(det.get("y2", 0), height),
                "score": round(score, 3),
                "class_id": class_id,
            }
            # Nhãn/ id theo dõi chỉ gửi khi có, cho gói gọn.
            name = entry.get("name")
            if name:
                box["label"] = name
            tid = det.get("tracker_id")
            if tid is not None and int(tid) >= 0:
                box["tid"] = int(tid)

            # POSE: engine gửi keypoints dạng bộ ba (x, y, score) PHẲNG,
            # toạ độ full-res. Chuẩn hoá x/y như box; giữ nguyên dạng phẳng
            # (gọn hơn mảng object ~3 lần) và bỏ điểm score<=0 bằng cách
            # đặt score 0 — client tự bỏ qua.
            kps = det.get("keypoints") or []
            if kps:
                flat = []
                for k in range(0, len(kps) - 2, 3):
                    flat.append(norm(kps[k], width))
                    flat.append(norm(kps[k + 1], height))
                    flat.append(round(float(kps[k + 2]), 2))
                if flat:
                    box["kps"] = flat

            # MASK phân vùng: engine gửi lưới bit GxG dạng HEX phủ đúng bbox.
            # Chuyển thẳng sang client, không dựng polygon ở đây — 256 ký tự
            # rẻ hơn nhiều so với tìm biên rồi đơn giản hoá phía server.
            mask_hex = det.get("mask")
            if mask_hex:
                box["mask"] = mask_hex
                box["mask_grid"] = int(det.get("maskGrid") or 32)

            boxes.append(box)

        return boxes, width, height

    def _publish_live_boxes(self, camera_id, job_id, ai_type, meta, boxes,
                            width, height, polygons=None) -> None:
        """Hot path. Đẩy khung phát hiện cho người đang xem TRỰC TIẾP.

        Khác `_stash_debug_frame`: KHÔNG đòi có JPEG. Engine C++ chỉ mã hoá ảnh
        khi khung có ít nhất một phát hiện, nên nếu bắt buộc có ảnh thì khung
        "không thấy gì" sẽ bị bỏ qua và lớp phủ trên màn hình sẽ ĐỨNG YÊN mãi ở
        khung cuối. Ở đây khung rỗng vẫn phải gửi để client xoá đi."""
        if not live_detection_broadcaster.has_clients(camera_id):
            return
        try:
            def norm(value, size):
                return round(min(1.0, max(0.0, float(value) / size)), 4)

            now = time.time()
            payload = {
                "camera_id": str(camera_id),
                "job_id": str(job_id),
                "ai_type": ai_type,
                "seq": meta.get("seq", 0),
                "ts": now,
                "width": int(width),
                "height": int(height),
                "boxes": boxes,
            }

            # Vùng chỉ kèm theo mỗi ZONE_RESEND_SEC giây. KHÔNG có khoá `zones`
            # nghĩa là "giữ nguyên vùng cũ"; có mà rỗng nghĩa là "không có
            # vùng nào" (job chạy toàn khung) — hai chuyện khác nhau.
            zkey = (str(camera_id), str(job_id))
            if now - self._zone_sent_at.get(zkey, 0.0) >= ZONE_RESEND_SEC:
                self._zone_sent_at[zkey] = now
                zones = []
                for poly in (polygons or []):
                    # None = vùng ảo TOÀN KHUNG (không có polygon thật) — không
                    # vẽ, vì vẽ ra chỉ là cái khung viền quanh cả màn hình.
                    if poly is None:
                        continue
                    pts = [
                        [norm(px, width), norm(py, height)]
                        for px, py in poly.tolist()
                    ]
                    if len(pts) >= 3:
                        zones.append(pts)
                payload["zones"] = zones

            live_detection_broadcaster.publish(payload)
        except Exception as exc:  # không bao giờ để lớp phủ làm chết vòng nhận
            print(f"live boxes publish skipped: {exc}", file=sys.stderr)

    def _drain_invalidations(self) -> None:
        with self._invalidate_lock:
            if not self._pending_invalidations:
                return
            pending = self._pending_invalidations
            self._pending_invalidations = set()
        for camera_id, job_id in pending:
            cam_jobs = self.process_ai.get(camera_id)
            if cam_jobs is None:
                continue
            if job_id is None:
                self.process_ai.pop(camera_id, None)
                # Also drop every dedup entry for this camera so any
                # leftover orphan publishes after a camera delete log
                # exactly once and not silently.
                self._unknown_job_logged = {
                    (c, j) for (c, j) in self._unknown_job_logged
                    if c != camera_id
                }
            else:
                cam_jobs.pop(job_id, None)
                if not cam_jobs:
                    self.process_ai.pop(camera_id, None)
                self._unknown_job_logged.discard((camera_id, job_id))

    def recv_exact(self, sock, n):
        chunks, got = [], 0
        while got < n:
            chunk = sock.recv(n - got)
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def recv_message(self, sock):
        header = self.recv_exact(sock, 4)
        if header is None:
            return None
        body = self.recv_exact(sock, struct.unpack(">I", header)[0])
        if body is None:
            return None
        json_len = struct.unpack(">I", body[:4])[0]
        meta = json.loads(body[4:4 + json_len].decode("utf-8"))
        return meta, body[4 + json_len:]

    def _connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.url_socket)
            print(f"Connected to AI engine at {self.url_socket}")
            return True
        except OSError as exc:
            print(f"Cannot connect to {self.url_socket}: {exc}", file=sys.stderr)
            self._close_sock()
            return False

    async def _load_ai_config(self, camera_id, job_id):
        async with self._session_factory() as db:
            return await AIRepository.get_by_camera_and_job(db, camera_id, job_id)

    async def _recv_loop(self):
        while self.running:
            try:
                message = await asyncio.to_thread(self.recv_message, self.sock)
            except OSError:
                return
            if message is None:
                print("AI engine closed the connection", file=sys.stderr)
                return

            meta, full_jpeg = message
            camera_id = meta.get("cameraId")
            job_id = meta.get("jobId")
            orig_width = meta.get("width")
            orig_height = meta.get("height")

            # Engine hạ tần suất encode JPEG (xem AiJob::kDetJpegMinGapMs) nên
            # nhiều khung tới KHÔNG kèm ảnh. Giữ ảnh đầy mới nhất theo camera và
            # thay vào khi khung hiện tại rỗng, để sự kiện (entered_zone...) luôn
            # có ảnh dù rơi đúng khung bị bỏ encode. Ảnh trễ ≤ ~kDetJpegMinGapMs.
            if full_jpeg:
                self._last_full_jpeg[camera_id] = full_jpeg
            else:
                full_jpeg = self._last_full_jpeg.get(camera_id) or full_jpeg

            # Pick up any API-driven config changes before deciding whether
            # the existing cached state is still valid.
            self._drain_invalidations()

            if camera_id not in self.process_ai or job_id not in self.process_ai[camera_id]:
                ai_config = await self._load_ai_config(camera_id, job_id)
                if ai_config is None:
                    key = (camera_id, job_id)
                    if key not in self._unknown_job_logged:
                        self._unknown_job_logged.add(key)
                        print(
                            f"AI config not found for camera_id={camera_id}, "
                            f"job_id={job_id} — likely an orphan C++ job. "
                            f"Suppressing further messages for this pair.",
                            file=sys.stderr,
                        )
                    continue
                # Resolved (config exists now) — clear the dedup flag so a
                # future deletion would re-emit one warning.
                self._unknown_job_logged.discard((camera_id, job_id))
                polygons, ids_in_zone, exit_pending, entered_at, dwell_alerted = \
                    ProcessAiHepper.prepare_zones(ai_config.polygons)
                tracker = ProcessAiHepper.init_tracker(
                    tracker_type=ai_config.tracker,
                    threshold=ai_config.primary_conf,
                    fps=ai_config.fps,
                )
                # Zone-exit grace must match how long the tracker keeps a
                # lost track's id alive. If it's shorter, a brief occlusion
                # makes the zone fire exited_zone (clearing the per-tracker
                # dedup state) while the tracker still holds the id — so the
                # object re-enters under the same id and a duplicate event is
                # written. Sizing the grace to the lost buffer closes that.
                exit_grace = ProcessAiHepper.lost_buffer_frames(
                    ai_config.tracker, ai_config.fps,
                )
                service_ai = ProcessAiHepper.get_service_ai(ai_config.type)
                self.process_ai.setdefault(camera_id, {})[job_id] = {
                    "tracker": tracker,
                    "polygons": polygons,
                    "ids_in_zone": ids_in_zone,
                    "exit_pending": exit_pending,
                    "entered_at": entered_at,
                    "dwell_alerted": dwell_alerted,
                    "fps": ai_config.fps,
                    "exit_grace": exit_grace,
                    "overlap_threshold": ai_config.overlap_threshold,
                    "dwell_seconds": ai_config.dwell_seconds or 0,
                    "service_ai": service_ai,
                    # Classes this service wants tracked (None = all). Only
                    # gates the tracker input; meta still carries every class.
                    "track_class_ids": ProcessAiHepper.get_track_class_ids(service_ai),
                    # Cosmetic per-class name/color for the debug overlay only.
                    "class_meta": ProcessAiHepper.get_class_meta(service_ai),
                    "secondary_conf": ai_config.secondary_conf,
                    "primary_conf": ai_config.primary_conf,
                    "ai_type": ai_config.type,
                    # Ghi khung phát hiện xuống DB để xem lại / tìm theo
                    # vùng. Mặc định TẮT.
                    "save_detections": bool(getattr(ai_config, "save_detections", False)),
                    # Free-form per-config JSON forwarded to the service hooks.
                    # Falls back to {} for rows saved before the column existed.
                    "extra_data": ai_config.extra_data if ai_config.extra_data is not None else {},
                }
            else:
                state = self.process_ai[camera_id][job_id]
                tracker = state["tracker"]
                polygons = state["polygons"]
                exit_pending = state["exit_pending"]
                ids_in_zone = state["ids_in_zone"]
                entered_at = state["entered_at"]
                dwell_alerted = state["dwell_alerted"]
                fps = state["fps"]
                exit_grace = state.get("exit_grace", fps)
                overlap_threshold = state["overlap_threshold"]
                dwell_seconds = state["dwell_seconds"]
                service_ai = state["service_ai"]
                secondary_conf = state["secondary_conf"]
                primary_conf = state.get("primary_conf", 0.3)
                ai_type = state.get("ai_type")
                save_detections = state.get("save_detections", False)
                track_class_ids = state.get("track_class_ids")
                class_meta = state.get("class_meta")
                extra_data = state.get("extra_data") or {}

                detections = ProcessAiHepper.to_sv_detections(meta.get("detections", []))
                # Tracker sees only the service's classes; `meta` is left
                # whole, so the debug overlay and event payloads still show
                # every class in the frame — just untracked.
                detections = ProcessAiHepper.filter_track_classes(
                    detections, track_class_ids,
                )
                # Off the loop: with BoTSORT this decodes the full JPEG and
                # runs optical-flow CMC (tens of ms on the RK3588 CPU); run
                # inline it would starve the fire-and-forget tasks (face
                # match, persist, websocket pushes) that share this loop.
                detections = await asyncio.to_thread(
                    ProcessAiHepper.update_tracker,
                    tracker, detections, full_jpeg, ai_type=ai_type,
                )
                if detections.tracker_id is None or len(detections) == 0:
                    # Nothing tracked this frame — but do NOT skip the zone
                    # bookkeeping below. Walking out of frame is the most
                    # common way to leave a zone, and it produces exactly
                    # this: zero detections. Returning early here would
                    # leave the tracker id in ids_in_zone forever, so its
                    # exit_pending never counts up and exited_zone never
                    # fires. Fall through with an empty set instead: every
                    # id currently in a zone is simply "not seen", which is
                    # what the exit grace is there to time out.
                    #
                    # Push the raw frame to any debug subscriber first so
                    # they can still see the model's pre-tracker output
                    # (helps spot tracker over-filtering). Cheap no-op when
                    # nobody's watching.
                    self._stash_debug_frame(camera_id, job_id, meta, full_jpeg, polygons,
                                            primary_conf, overlap_threshold, class_meta)
                    detections = ProcessAiHepper.empty_tracked_detections()
                else:
                    detections = detections[detections.tracker_id >= 0]

                    raw_dets = meta.get("detections", [])
                    if len(detections) and raw_dets:
                        raw_xyxy = np.array(
                            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in raw_dets],
                            dtype=np.float32,
                        )
                        for i in range(len(detections)):
                            box = detections.xyxy[i]
                            for j in range(len(raw_dets)):
                                if np.array_equal(raw_xyxy[j], box):
                                    raw_dets[j]["tracker_id"] = int(detections.tracker_id[i])
                                    break

                    # Stash with tracker_ids tagged onto raw_dets, so the
                    # MJPEG overlay can show them.
                    self._stash_debug_frame(camera_id, job_id, meta, full_jpeg, polygons,
                                            primary_conf, overlap_threshold, class_meta)

                # Khung phát hiện dùng cho HAI việc: lớp phủ trực tiếp và ghi
                # xuống DB để xem lại. Đặt SAU cả hai nhánh trên (và ngoài
                # chúng) để MỌI khung đều được xử lý — kể cả khung không phát
                # hiện gì, vốn là tín hiệu để client xoá khung cũ. Lọc theo
                # `primary_conf` nên khung vẽ ra đúng bằng khung hệ thống thực
                # sự theo dõi.
                #
                # Dựng box CHỈ KHI có người cần: không ai xem live mà cũng
                # không bật ghi thì bỏ qua hẳn (đây là mỗi khung của mọi camera).
                if save_detections or live_detection_broadcaster.has_clients(camera_id):
                    live_boxes, lw, lh = self._build_boxes(meta, class_meta, primary_conf)
                    if lw > 0:
                        self._publish_live_boxes(camera_id, job_id, ai_type, meta,
                                                 live_boxes, lw, lh, polygons)
                        if save_detections:
                            detection_slice_writer.add_frame(
                                camera_id, job_id, ai_type, live_boxes,
                            )

                # Ghi các lát đã chốt xuống DB theo MẺ, chạy nền: một cú ghi DB
                # chậm không được phép làm nghẽn luồng xử lý ảnh.
                if save_detections:
                    now_s = time.time()
                    if now_s - self._last_slice_flush >= 2.0:
                        self._last_slice_flush = now_s
                        asyncio.create_task(
                            detection_slice_writer.flush(self._session_factory)
                        )

                now = time.time()
                for zone_idx, polygon in enumerate(polygons):
                    if polygon is None:
                        in_zone_mask = np.ones(len(detections.xyxy), dtype=bool)
                    else:
                        in_zone_mask = np.array(
                            [ProcessAiHepper.bbox_in_zone(
                                bbox, polygon, overlap_threshold)
                             for bbox in detections.xyxy],
                            dtype=bool,
                        )
                    current_ids = set(detections.tracker_id[in_zone_mask].tolist())

                    for tid in current_ids:
                        exit_pending[zone_idx].pop(tid, None)
                        if tid not in ids_in_zone[zone_idx]:
                            # print(f"ID {tid} ENTERED zone {zone_idx}")
                            if service_ai is not None and hasattr(service_ai, "entered_zone"):
                                service_ai.entered_zone(tid, meta, full_jpeg, now, secondary_conf, extra_data, zone_idx)
                            ids_in_zone[zone_idx].add(tid)
                            entered_at[zone_idx][tid] = now
                        elif dwell_seconds > 0 and tid not in dwell_alerted[zone_idx]:
                            if now - entered_at[zone_idx].get(tid, now) >= dwell_seconds:
                                # print(f"ID {tid} STAYED in zone {zone_idx} for {dwell_seconds}s")
                                if service_ai is not None and hasattr(service_ai, "dwell_alert"):
                                    service_ai.dwell_alert(tid, meta, full_jpeg, now, secondary_conf, extra_data, zone_idx)
                                dwell_alerted[zone_idx].add(tid)
                        else:
                            if service_ai is not None and hasattr(service_ai, "in_the_area"):
                                service_ai.in_the_area(tid, meta, full_jpeg, now, secondary_conf, extra_data, zone_idx)

                    for tid in list(ids_in_zone[zone_idx] - current_ids):
                        exit_pending[zone_idx][tid] = exit_pending[zone_idx].get(tid, 0) + 1
                        if exit_pending[zone_idx][tid] >= exit_grace:
                            # print(f"ID {tid} EXITED zone {zone_idx}")
                            if service_ai is not None and hasattr(service_ai, "exited_zone"):
                                service_ai.exited_zone(tid, meta, full_jpeg, now, secondary_conf, extra_data, zone_idx)
                            ids_in_zone[zone_idx].discard(tid)
                            exit_pending[zone_idx].pop(tid, None)
                            entered_at[zone_idx].pop(tid, None)
                            dwell_alerted[zone_idx].discard(tid)

    def _close_sock(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    async def _run(self):
        engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
        self._session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        self.running = True
        try:
            while self.running:
                if self._connect():
                    try:
                        await self._recv_loop()
                    finally:
                        self._close_sock()
                        self.process_ai.clear()
                if self.running:
                    await asyncio.sleep(self.reconnect_delay)
        finally:
            await engine.dispose()

    def start(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(f"process_ai_service crashed: {exc}", file=sys.stderr)
        return 0

    def stop(self):
        self.running = False
        self._close_sock()


process_ai_service = ProcessAiService()
