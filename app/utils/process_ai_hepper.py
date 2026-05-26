import datetime
import json
import os

import cv2
import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker

from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.face_recognition_service import face_recognition_service
from app.services.plate_recognition_service import plate_recognition_service


class ProcessAiHepper:
    @staticmethod
    def init_tracker(tracker_type="bytetrack", threshold=0.25, fps=15):
        # "bytetrack" | "botsort"
        # high_conf_det_threshold must be <= threshold, otherwise no detection
        # ever counts as "high-confidence" and no new track can be spawned.
        high_conf = max(0.1, threshold - 0.1)
        if tracker_type == "botsort":
            tracker = BoTSORTTracker(
                track_activation_threshold=threshold,
                lost_track_buffer=30,
                frame_rate=fps,
                minimum_consecutive_frames=1,
                high_conf_det_threshold=high_conf,
            )
        else:
            tracker = ByteTrackTracker(
                track_activation_threshold=threshold,
                lost_track_buffer=30,
                frame_rate=fps,
                minimum_consecutive_frames=1,
                high_conf_det_threshold=high_conf,
            )
        return tracker

    @staticmethod
    def bbox_zone_overlap(bbox, polygon):
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = polygon - np.array([x1, y1])
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
        return float(np.count_nonzero(mask)) / (w * h)

    @staticmethod
    def prepare_zones(polygon_points):
        polygons = [np.array(p, dtype=np.int32) for p in json.loads(polygon_points)]
        if not polygons:
            polygons = [None]  # full-frame virtual zone

        ids_in_zone = [set() for _ in polygons]
        exit_pending = [{} for _ in polygons]
        entered_at = [{} for _ in polygons]
        dwell_alerted = [set() for _ in polygons]
        return polygons, ids_in_zone, exit_pending, entered_at, dwell_alerted

    @staticmethod
    def update_tracker(tracker, detections, full_jpeg):
        if isinstance(tracker, BoTSORTTracker):
            frame = ProcessAiHepper.decode_frame(full_jpeg)
            return tracker.update(detections, frame)
        return tracker.update(detections)

    @staticmethod
    def decode_frame(full_jpeg):
        if not full_jpeg:
            return None
        return cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)

    @staticmethod
    def to_sv_detections(raw_detections):
        if not raw_detections:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=int),
            )
        xyxy = np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in raw_detections],
            dtype=np.float32,
        )
        confidence = np.array([d["score"] for d in raw_detections], dtype=np.float32)
        class_id = np.array([d["classId"] for d in raw_detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @staticmethod
    def get_service_ai(type: str):
        if type == TypeConfigAiEnum.FACE_RECOGNITION.value:
            return face_recognition_service
        elif type == TypeConfigAiEnum.PLATE_RECOGNITION.value:
            return plate_recognition_service
        else:
            return None

    @staticmethod
    def save_image(save_dir, full_jpeg, meta, detection):
        if not full_jpeg:
            return None, None
        date = datetime.date.today().isoformat()
        folder = os.path.join(save_dir, str(meta["cameraId"]), date)
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, f"{int(meta['seq']):010d}")

        full_path = f"{stem}_full.jpg"
        crop_path = None
        with open(full_path, "wb") as fp:
            fp.write(full_jpeg)

        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            x1 = max(0, min(int(detection["x1"]), w - 1))
            y1 = max(0, min(int(detection["y1"]), h - 1))
            x2 = max(x1 + 1, min(int(detection["x2"]), w))
            y2 = max(y1 + 1, min(int(detection["y2"]), h))
            crop_path = f"{stem}_det.jpg"
            cv2.imwrite(crop_path, img[y1:y2, x1:x2])
        return full_path, crop_path
