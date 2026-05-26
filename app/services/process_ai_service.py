import asyncio
import json
import socket
import struct
import sys
import time

import numpy as np

from app.services.camera_service import camera_service
from app.utils.process_ai_hepper import ProcessAiHepper


class ProcessAiService:
    def __init__(self, url_socket="/tmp/ai_engine.sock", reconnect_delay=2.0):
        self.url_socket = url_socket
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.sock = None
        self.process_ai = {}

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

    def _recv_loop(self):
        while self.running:
            try:
                message = self.recv_message(self.sock)
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

            if camera_id not in self.process_ai or job_id not in self.process_ai[camera_id]:
                ai_config = asyncio.run(camera_service.get_ai_config(camera_id, job_id))
                if ai_config is None:
                    print(
                        f"AI config not found for camera_id={camera_id}, job_id={job_id}",
                        file=sys.stderr,
                    )
                    continue
                polygons, ids_in_zone, exit_pending, entered_at, dwell_alerted = \
                    ProcessAiHepper.prepare_zones(ai_config.polygons)
                tracker = ProcessAiHepper.init_tracker(tracker_type=ai_config.tracker,
                                                       threshold=ai_config.primary_conf, fps=ai_config.fps)
                self.process_ai.setdefault(camera_id, {})[job_id] = {
                    "tracker": tracker,
                    "polygons": polygons,
                    "ids_in_zone": ids_in_zone,
                    "exit_pending": exit_pending,
                    "entered_at": entered_at,
                    "dwell_alerted": dwell_alerted,
                    "fps": ai_config.fps,
                    "overlap_threshold": ai_config.overlap_threshold,
                    "dwell_seconds": ai_config.dwell_seconds or 0,
                    "service_ai": ProcessAiHepper.get_service_ai(ai_config.type),
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
                overlap_threshold = state["overlap_threshold"]
                dwell_seconds = state["dwell_seconds"]
                service_ai = state["service_ai"]

                detections = ProcessAiHepper.to_sv_detections(meta.get("detections", []))

                detections = detections[detections.tracker_id >= 0]

                now = time.time()
                for zone_idx, polygon in enumerate(polygons):
                    if polygon is None:
                        in_zone_mask = np.ones(len(detections.xyxy), dtype=bool)
                    else:
                        in_zone_mask = np.array(
                            [ProcessAiHepper.bbox_zone_overlap(bbox, polygon) >= overlap_threshold for bbox in
                             detections.xyxy],
                            dtype=bool,
                        )
                    current_ids = set(detections.tracker_id[in_zone_mask].tolist())

                    for tid in current_ids:
                        exit_pending[zone_idx].pop(tid, None)
                        if tid not in ids_in_zone[zone_idx]:
                            print(f"ID {tid} ENTERED zone {zone_idx}")
                            ids_in_zone[zone_idx].add(tid)
                            entered_at[zone_idx][tid] = now
                        elif dwell_seconds > 0 and tid not in dwell_alerted[zone_idx]:
                            if now - entered_at[zone_idx].get(tid, now) >= dwell_seconds:
                                print(f"ID {tid} STAYED in zone {zone_idx} for {dwell_seconds}s")
                                dwell_alerted[zone_idx].add(tid)

                    for tid in list(ids_in_zone[zone_idx] - current_ids):
                        exit_pending[zone_idx][tid] = exit_pending[zone_idx].get(tid, 0) + 1
                        if exit_pending[zone_idx][tid] >= fps:
                            print(f"ID {tid} EXITED zone {zone_idx}")
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

    def start(self):
        self.running = True
        while self.running:
            if self._connect():
                try:
                    self._recv_loop()
                finally:
                    self._close_sock()
            if self.running:
                time.sleep(self.reconnect_delay)
        return 0

    def stop(self):
        self.running = False
        self._close_sock()
