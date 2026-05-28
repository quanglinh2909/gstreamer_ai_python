"""Draw AI detection results on top of a JPEG frame.

Used by the debug MJPEG endpoint. The function decodes the JPEG, paints
detection boxes / zone polygons / tracker IDs / per-task labels (plate
text from OCR children, identity name when wired in later), then
re-encodes back to JPEG. CPU-bound; the caller should run it via
`asyncio.to_thread` so it doesn't stall the FastAPI loop."""

from __future__ import annotations

import cv2
import numpy as np

from app.utils.plate_recognition_hepper import detect_plate_from_children
from app.utils.process_ai_hepper import ProcessAiHepper

_ZONE_COLOR = (0, 255, 255)       # yellow polygons — distinct from box colors
_IN_ZONE_COLOR = (0, 200, 0)      # green box: detection is inside a drawn zone
_OUT_ZONE_COLOR = (0, 0, 255)     # red box: detection outside every zone
_KP_COLOR = (255, 0, 255)         # magenta — face landmarks
_LABEL_TEXT_COLOR = (255, 255, 255)  # white text on the coloured background
_FOOTER_COLOR = (0, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.2
_FONT_THICKNESS = 3


def _det_in_any_zone(bbox, polygons) -> bool:
    """True when the detection counts as 'in zone' by the same rule the
    recv loop uses (bottom-centre `pointPolygonTest` per polygon).
    A None polygon is the sentinel for "full-frame virtual zone" so any
    bbox is considered inside it."""
    if not polygons:
        return True
    for poly in polygons:
        if poly is None:
            return True
        if ProcessAiHepper.bbox_in_zone(bbox, poly):
            return True
    return False


def _draw_label(img, text, x, y, bg_color):
    """Draw `text` with a filled background rectangle so it stays
    readable over any underlying image. (x, y) is the top-left of the
    label band — the band extends downward by the text height."""
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
    pad = 4
    band_h = th + baseline + pad * 2
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x1 + tw + pad * 2)
    y2 = min(img.shape[0], y1 + band_h)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    text_org = (x1 + pad, y1 + th + pad)
    cv2.putText(img, text, text_org, _FONT, _FONT_SCALE,
                _LABEL_TEXT_COLOR, _FONT_THICKNESS, cv2.LINE_AA)


def draw_overlay(meta: dict, full_jpeg: bytes, polygons) -> bytes:
    # Defensive: process_ai_service already filters empty jpegs out, but
    # a race or future code path could still reach us with no bytes.
    # `imdecode` asserts on an empty buffer, which would tear the whole
    # MJPEG stream down — return empty so the router skips this tick.
    if not full_jpeg:
        return b""
    img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return b""

    # Zones (skip the sentinel None used for "full-frame virtual zone")
    if polygons:
        for poly in polygons:
            if poly is None:
                continue
            pts = np.asarray(poly, dtype=np.int32)
            cv2.polylines(img, [pts], True, _ZONE_COLOR, 2)

    for det in meta.get("detections", []):
        x1 = int(det.get("x1", 0))
        y1 = int(det.get("y1", 0))
        x2 = int(det.get("x2", 0))
        y2 = int(det.get("y2", 0))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        # Colour by zone membership — green inside a zone, red outside.
        # Same rule the recv loop uses for entered/exited events, so the
        # visual matches what the AI pipeline actually counts.
        color = _IN_ZONE_COLOR if _det_in_any_zone(bbox, polygons) else _OUT_ZONE_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Compact label: tracker_id (when assigned), class, score, plate.
        parts = []
        tid = det.get("tracker_id")
        if tid is not None and tid >= 0:
            parts.append(f"id={int(tid)}")
        parts.append(f"cls={det.get('classId', '?')}")
        parts.append(f"{float(det.get('score', 0.0)):.2f}")
        children = det.get("children")
        if children:
            try:
                plate = detect_plate_from_children(children, 0.3)
                if plate:
                    parts.append(plate)
            except Exception:
                pass  # OCR helper is best-effort here

        label = " ".join(parts)
        # Anchor the label band just above the box; flip below the box
        # when the bbox sits too close to the top edge to fit the band.
        (_, band_th), band_baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS,
        )
        band_total_h = band_th + band_baseline + 8
        if y1 - band_total_h >= 0:
            ly = y1 - band_total_h
        else:
            ly = y2  # spill below the box
        _draw_label(img, label, x1, ly, color)

        # Face pose keypoints — flat (x, y, score) triples.
        kps = det.get("keypoints") or []
        for k in range(0, len(kps) - 2, 3):
            if kps[k + 2] > 0:
                cv2.circle(img, (int(kps[k]), int(kps[k + 1])),
                           4, _KP_COLOR, -1)

    footer = (
        f"cam={meta.get('cameraId','')} job={meta.get('jobId','')} "
        f"seq={meta.get('seq','')} det={len(meta.get('detections', []))}"
    )
    # Footer also gets the readable filled band, anchored bottom-left.
    (_, ft_th), ft_baseline = cv2.getTextSize(
        footer, _FONT, _FONT_SCALE, _FONT_THICKNESS,
    )
    _draw_label(img, footer, 8, img.shape[0] - (ft_th + ft_baseline + 10),
                (50, 50, 50))

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else full_jpeg
