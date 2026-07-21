"""Shared plumbing for the per-camera AI services.

Every AI service (restricted area, face recognition, plate recognition,
face mask) is driven by the same detection hooks from `process_ai_service`
and — when it persists an event — runs the same routine: find the
detection carrying a tracker id, decode the frame, cut a fixed-size crop,
then write the full frame + crop under
`/uploads/<folder>/<cameraId>/<date>/`.

Only three things actually differ between services: the crop geometry,
the upload subfolder, and the filename suffix. Those are class attributes
(and one small override) here instead of four near-identical copies of
the same code.
"""

import datetime
import os

import cv2
import numpy as np

from app.utils.image_crop import fixed_size_crop

# <repo>/uploads — services import this from here so every one of them
# resolves the same directory. Re-exported by the service modules that
# used to define it locally, so existing imports keep working.
UPLOADS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
)


class AIServiceBase:
    """Mixin providing the detection-hook helpers shared by all AI services.

    Subclasses override the crop constants and `EVENT_FOLDER`; services
    that never save images (e.g. face mask) inherit only `_find_parent`.
    """

    # Subfolder under /uploads for this service's event images. Required
    # only by services that call `_save_images_blocking`.
    EVENT_FOLDER = None

    # Crop geometry. `CROP_PAD_*` is outward padding as a ratio of the
    # bbox width/height; `CROP_OUTPUT_*` is the fixed output size (aspect
    # preserved, letterboxed with `CROP_PAD_COLOR`). `CROP_VERTICAL_BIAS`
    # is "center" or "below" — see `fixed_size_crop`.
    CROP_PAD_LEFT = 0.2
    CROP_PAD_RIGHT = 0.2
    CROP_PAD_TOP = 0.2
    CROP_PAD_BOTTOM = 0.2
    CROP_OUTPUT_W = 400
    CROP_OUTPUT_H = 480
    CROP_PAD_COLOR = 114  # neutral grey, matches YOLO letterboxing
    CROP_VERTICAL_BIAS = "center"

    @staticmethod
    def _find_parent(meta, tid):
        """The detection in this frame carrying `tid`, or None.

        `process_ai_service` tags tracker ids back onto the raw detection
        dicts, so this is how a hook gets from an id to its bbox/score."""
        for d in meta.get("detections", []):
            if d.get("tracker_id") == tid:
                return d
        return None

    @classmethod
    def _make_crop(cls, img, bx1, by1, bx2, by2):
        """Fixed-size crop around a bbox using this service's geometry."""
        return fixed_size_crop(
            img, bbox=(bx1, by1, bx2, by2),
            pad_lrtb=(cls.CROP_PAD_LEFT, cls.CROP_PAD_RIGHT,
                      cls.CROP_PAD_TOP, cls.CROP_PAD_BOTTOM),
            output_size=(cls.CROP_OUTPUT_W, cls.CROP_OUTPUT_H),
            pad_color=cls.CROP_PAD_COLOR,
            vertical_bias=cls.CROP_VERTICAL_BIAS,
        )

    @classmethod
    def _stem_suffix(cls, value):
        """Filename part after the frame seq. Tracker id by default;
        plate recognition overrides it with the sanitised plate text."""
        return str(int(value))

    @classmethod
    def _save_images_blocking(cls, full_jpeg, meta, parent, stem_value):
        """Write the full frame and its crop; return (full_url, crop_url).

        Returns None when there's nothing renderable (no bytes, undecodable
        JPEG, degenerate crop, failed write) so callers can skip persisting
        an event with dangling image paths. Blocking on purpose — callers
        run it through `asyncio.to_thread`."""
        if not full_jpeg:
            return None
        img = cv2.imdecode(np.frombuffer(full_jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        crop = cls._make_crop(
            img,
            float(parent.get("x1", 0.0)), float(parent.get("y1", 0.0)),
            float(parent.get("x2", 0.0)), float(parent.get("y2", 0.0)),
        )
        if crop is None:
            return None

        date = datetime.date.today().isoformat()
        folder_rel = os.path.join(cls.EVENT_FOLDER, str(meta["cameraId"]), date)
        folder_abs = os.path.join(UPLOADS_ROOT, folder_rel)
        os.makedirs(folder_abs, exist_ok=True)

        stem = f"{int(meta['seq']):010d}_{cls._stem_suffix(stem_value)}"
        full_abs = os.path.join(folder_abs, f"{stem}_full.jpg")
        crop_abs = os.path.join(folder_abs, f"{stem}_crop.jpg")

        with open(full_abs, "wb") as fp:
            fp.write(full_jpeg)
        if not cv2.imwrite(crop_abs, crop):
            return None
        return (
            f"/uploads/{folder_rel}/{stem}_full.jpg",
            f"/uploads/{folder_rel}/{stem}_crop.jpg",
        )
