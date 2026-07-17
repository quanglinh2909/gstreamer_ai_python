import datetime
import json
import os

import cv2
import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker

from app.enum.config_ai_enum import TypeConfigAiEnum
from app.services.face_recognition_service import face_recognition_service
from app.services.plate_recognition_service import plate_recognition_service
from app.services.restricted_area_service import restricted_area_service



class ProcessAiHepper:
    @staticmethod
    def lost_buffer_frames(tracker_type="ocsort", fps=15):
        """How many frames a tracker keeps a lost track alive (and thus the
        same tracker_id) before dropping it. The zone-exit grace in
        process_ai_service is sized off this so the zone never declares an
        object "exited" while the tracker still holds its id — otherwise a
        brief occlusion clears the per-tracker dedup state and the object
        re-enters under the *same* id, producing a duplicate event."""
        if tracker_type in ("botsort", "bytetrack"):
            return max(10, int(fps * 2))
        return max(15, int(fps * 3))  # ocsort default (~3s)

    @staticmethod
    def init_tracker(tracker_type="ocsort", threshold=0.25, fps=15):
        # "ocsort" (default, best for low fps) | "bytetrack" | "botsort"
        # high_conf_det_threshold must be <= threshold, otherwise no detection
        # ever counts as "high-confidence" and no new track can be spawned.
        high_conf = max(0.1, threshold - 0.1)
        # Buffer in *wall time*, not raw frames. Default 30 frames at the
        # tracker's nominal 30fps = 1s; at fps=5 that became 6s (too long,
        # zombie tracks linger and steal IDs) and at fps=30 only 1s (too
        # short, a one-frame detector miss kills the track). Target ~2s so
        # a brief drop in detection still resumes onto the same tracker_id.
        buffer = ProcessAiHepper.lost_buffer_frames(tracker_type, fps)

        if tracker_type == "botsort":
            return BoTSORTTracker(
                track_activation_threshold=threshold,
                lost_track_buffer=buffer,
                frame_rate=fps,
                minimum_consecutive_frames=1,
                high_conf_det_threshold=high_conf,
                minimum_iou_threshold_first_assoc=0.1,
                minimum_iou_threshold_second_assoc=0.3,
                enable_cmc=True,
            )
        if tracker_type == "bytetrack":
            return ByteTrackTracker(
                track_activation_threshold=threshold,
                lost_track_buffer=buffer,
                frame_rate=fps,
                minimum_consecutive_frames=1,
                minimum_iou_threshold=0.1,
                high_conf_det_threshold=high_conf,
            )

        # Default: OC-SORT. Observation-Centric SORT is purpose-built for
        # low-fps + occlusion + non-linear motion — the exact failure mode
        # here. ORU re-updates a Kalman track from its virtual trajectory
        # when it reappears after a miss (so the id survives a detector
        # gap), and OCM uses motion *direction* over delta_t frames to
        # associate when frame-to-frame IoU has already collapsed.
        #
        #  - lost_track_buffer ~3s: ORU needs the track kept alive long
        #    enough to re-acquire after an occlusion/gap.
        #  - delta_t = fps: one second of velocity history; long enough at
        #    low fps to estimate a stable direction.
        #  - minimum_iou_threshold 0.2: looser than the 0.3 default for the
        #    large per-frame displacement at low fps.
        return OCSORTTracker(
            lost_track_buffer=buffer,
            frame_rate=fps,
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.2,
            delta_t=max(3, int(fps)),
            high_conf_det_threshold=high_conf,
        )

    # Fraction of a bbox that must fall inside a zone before it counts as
    # "in". Mirrors the DTO / frontend default (UI shows 30, sent as 0.30)
    # and is the fallback when ai_config.overlap_threshold is NULL — the
    # column is nullable, so rows predating the setting still work.
    DEFAULT_OVERLAP_THRESHOLD = 0.30

    @staticmethod
    def bbox_zone_overlap(bbox, polygon):
        """Fraction of the bbox's area that lies inside the polygon, 0..1.

        Rasterises the polygon into a bbox-sized mask and counts covered
        pixels, so concave / multi-vertex zones are handled exactly.
        cv2.fillPoly clips to the mask, which is what bounds the count to
        the intersection."""
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return 0.0
        mask = np.zeros((h, w), dtype=np.uint8)
        shifted = polygon - np.array([x1, y1])
        cv2.fillPoly(mask, [shifted.astype(np.int32)], 255)
        return float(np.count_nonzero(mask)) / (w * h)

    @staticmethod
    def bbox_in_zone(bbox, polygon, overlap_threshold=None) -> bool:
        """Zone membership by area coverage: the detection is "in" once at
        least `overlap_threshold` of its bbox area falls inside the polygon.

        This is what the overlap_threshold knob on the UI drives — 0.30
        means "counts as inside once 30% of the object is in the zone".
        A threshold of 0 (or less) means any contact at all counts; without
        that special case `overlap >= 0` would be true for every detection
        in the frame, including ones nowhere near the zone.

        Note the tradeoff vs. the bottom-centre anchor test this replaced
        (the Frigate / DeepStream / supervision default): area coverage is
        judged purely in image space, so an object standing *behind* the
        zone but whose bbox column crosses it can register as inside.
        Draw zones with that in mind on cameras with a lot of depth."""
        if overlap_threshold is None:
            overlap_threshold = ProcessAiHepper.DEFAULT_OVERLAP_THRESHOLD
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
        overlap = ProcessAiHepper.bbox_zone_overlap(bbox, polygon)
        if overlap_threshold <= 0:
            return overlap > 0.0
        return overlap >= overlap_threshold

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
    def update_tracker(tracker, detections, full_jpeg, ai_type=None):
        """Feed detections into the tracker with per-AI-type bbox padding.

        Face boxes are tiny (head only); without inflating them into a
        head+shoulders region, IoU between frames collapses at low fps
        and the tracker spawns fresh ids. Plate boxes need a smaller
        inflate. Person body / restricted-area boxes are already big
        enough — inflating them makes adjacent people's boxes overlap
        ~80%, which is the opposite of what you want and causes the id
        jumps you'd see in a busy scene. Profile per ai_type.

        The tracker preserves input order, so tracker_id is copied back
        onto the original (unpadded) detections by index — downstream
        code keeps seeing the real bbox."""
        if len(detections) == 0:
            return detections
        padded = ProcessAiHepper._inflate_for_tracking(detections, ai_type)
        if isinstance(tracker, BoTSORTTracker):
            frame = ProcessAiHepper.decode_frame(full_jpeg)
            tracked = tracker.update(padded, frame)
        else:
            tracked = tracker.update(padded)
        if tracked.tracker_id is not None and len(tracked) == len(detections):
            detections.tracker_id = tracked.tracker_id
        return detections

    # Inflate profile per ai_type: (pad_l, pad_r, pad_t, pad_b) as ratios
    # of bbox width / height. Default for unknown types = no inflate.
    #
    # face_recognition  : tiny head bbox → expand to a head+shoulders
    #                     region so IoU stays high across motion at low fps.
    # plate_recognition : small wide bbox → modest pad covers a chunk of
    #                     the surrounding car. Side padding small so two
    #                     adjacent plates don't fuse.
    # restricted_area   : whole person body — only a *gentle* inflate
    #                     (~20%). Bigger would make adjacent people's
    #                     padded boxes overlap heavily and the tracker
    #                     would swap ids when they cross. Zero inflate
    #                     drops tracks at low fps + fast motion (raw
    #                     bbox IoU collapses to 0). 20% is the sweet
    #                     spot for typical entry / corridor cameras.
    _INFLATE_BY_TYPE = {
        "face_recognition":  (1.5, 1.5, 0.4, 3.0),
        "plate_recognition": (0.3, 0.3, 1.5, 1.5),
        "restricted_area":   (0.2, 0.2, 0.2, 0.2),
    }

    @staticmethod
    def _inflate_for_tracking(detections, ai_type=None):
        pad = ProcessAiHepper._INFLATE_BY_TYPE.get(
            ai_type, (0.0, 0.0, 0.0, 0.0),
        )
        pad_l, pad_r, pad_t, pad_b = pad
        if max(pad) == 0:
            return detections  # no-op fast path
        xyxy = detections.xyxy
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        inflated = xyxy.copy()
        inflated[:, 0] = xyxy[:, 0] - w * pad_l
        inflated[:, 1] = xyxy[:, 1] - h * pad_t
        inflated[:, 2] = xyxy[:, 2] + w * pad_r
        inflated[:, 3] = xyxy[:, 3] + h * pad_b
        return sv.Detections(
            xyxy=inflated.astype(np.float32),
            confidence=detections.confidence,
            class_id=detections.class_id,
        )

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
    def filter_track_classes(detections, class_ids):
        """Narrow the tracker's input to the classes a service cares about.

        This is deliberately *only* applied to what goes into the tracker.
        `meta["detections"]` keeps every class the model found, so the debug
        overlay and any downstream consumer still see the whole frame — only
        the filtered classes come back carrying a tracker_id.

        Contrast with AIJobSpec.class_filter, which drops classes inside the
        C++ engine before the message is ever sent: that one removes them
        from meta too. Use this when you want zone/tracking logic on one
        class but still want to see the rest.

        Falsy `class_ids` means "track every class"."""
        if not class_ids or len(detections) == 0 or detections.class_id is None:
            return detections
        return detections[np.isin(detections.class_id, list(class_ids))]

    @staticmethod
    def get_track_class_ids(service_ai):
        """Per-service tracking class whitelist, or None to track all."""
        return getattr(service_ai, "TRACK_CLASS_IDS", None)

    @staticmethod
    def get_service_ai(type: str):
        if type == TypeConfigAiEnum.FACE_RECOGNITION.value:
            return face_recognition_service
        elif type == TypeConfigAiEnum.PLATE_RECOGNITION.value:
            return plate_recognition_service
        elif type == TypeConfigAiEnum.RESTRICTED_AREA.value:
            return restricted_area_service
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
