"""Offline face-embedding pipeline used for identity registration.

Mirrors the proven /home/orangepi/Documents/test/ai_result_face.py reference:
    image -> YOLO pose -> 5 landmarks -> AdaFace align_face -> RKNN AdaFace ->
    L2-normalized embedding.

Models are loaded lazily on first call and kept as singletons (loading the
RKNN model costs hundreds of ms). All work is CPU/NPU-bound; callers should
wrap calls in `asyncio.to_thread` to avoid blocking the event loop.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.utils.align_face import align_face as _align_face_fn


@dataclass
class FaceExtractResult:
    embedding: List[float]                       # L2-normalized, len = embedding dim
    bbox: Tuple[float, float, float, float]      # x1, y1, x2, y2 in original image coords
    score: float                                 # detector confidence


_lock = threading.Lock()
_detector = None
_rknn = None


def _load_yolo(path: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics not installed. Install with: pip install ultralytics"
        ) from exc
    return YOLO(path, task="pose")


def _load_rknn(path: str):
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError(
            "rknnlite not installed. Install rknn-toolkit-lite2 for your platform."
        ) from exc
    rknn = RKNNLite()
    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {path} (ret={ret})")
    ret = rknn.init_runtime()
    if ret != 0:
        rknn.release()
        raise RuntimeError(f"Failed to init RKNN runtime for {path} (ret={ret})")
    return rknn


def _ensure_loaded() -> None:
    global _detector, _rknn
    if _detector is not None and _rknn is not None:
        return
    with _lock:
        if _detector is None:
            _detector = _load_yolo(settings.FACE_DETECTOR_MODEL_PATH)
            print(f"[face_embedder] loaded detector: {settings.FACE_DETECTOR_MODEL_PATH}",
                  file=sys.stderr)
        if _rknn is None:
            _rknn = _load_rknn(settings.FACE_EMBEDDING_MODEL_PATH)
            print(f"[face_embedder] loaded AdaFace: {settings.FACE_EMBEDDING_MODEL_PATH}",
                  file=sys.stderr)


def _l2_normalize(vec: np.ndarray) -> Optional[List[float]]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return None
    return (vec / norm).astype(np.float32).tolist()


def extract_face_embedding(image_bytes: bytes) -> Optional[FaceExtractResult]:
    """Detect → pick largest face → align (MTCNN-style 5pt similarity transform)
    → AdaFace embed → L2-normalize.

    Returns None when no face is found or any pipeline stage fails."""
    _ensure_loaded()

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    results = _detector.predict(image, conf=settings.FACE_DETECT_CONF, verbose=False)

    best_box = None
    best_kp = None
    best_area = -1.0
    best_score = 0.0
    for result in results:
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        kp_data = keypoints.data.cpu().numpy()
        for box, conf, kp in zip(xyxy, confs, kp_data):
            if len(kp) < 5:
                continue
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            if area > best_area:
                best_area = area
                best_box = box
                best_kp = kp
                best_score = float(conf)

    if best_kp is None or best_box is None:
        return None

    aligned_bgr = _align_face_fn(image, best_kp)
    if aligned_bgr is None:
        return None

    input_tensor = np.expand_dims(aligned_bgr.astype(np.float32), axis=0)
    outputs = _rknn.inference(inputs=[input_tensor], data_format="nhwc")
    if not outputs:
        return None

    vec = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
    embedding = _l2_normalize(vec)
    if embedding is None:
        return None

    return FaceExtractResult(
        embedding=embedding,
        bbox=(float(best_box[0]), float(best_box[1]),
              float(best_box[2]), float(best_box[3])),
        score=best_score,
    )
