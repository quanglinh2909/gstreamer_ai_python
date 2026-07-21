"""Fixed-size aspect-preserving crop with edge-shift compensation.

When the desired crop window extends past the source frame, the naive
letterbox approach fills the missing area with grey — leaving an ugly
gap on one side. This helper *shifts* the window inward toward the
opposite edge first ("top out of bounds → take more from below"), so
the saved image always shows real pixels when there's room for them.
Grey letterbox is reserved for the case where the source frame itself
is smaller than the requested target size — there's literally nothing
else to pull from."""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np

_VerticalBias = Literal["center", "below"]


def fixed_size_crop(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_lrtb: Tuple[float, float, float, float],
    output_size: Tuple[int, int],
    pad_color: int = 114,
    vertical_bias: _VerticalBias = "center",
) -> np.ndarray | None:
    """Return a fixed-size BGR crop centred on `bbox`.

    bbox        — (x1, y1, x2, y2) in source coords.
    pad_lrtb    — outward padding as ratio of bbox width/height:
                  (left, right, top, bottom).
    output_size — (target_w, target_h). Aspect is preserved.
    vertical_bias — when the padded box is wider than target aspect the
                    extra height is added: 'center' = symmetric, 'below'
                    = anchored to the box top (extra space goes down,
                    natural for face → body crops)."""
    h, w = img.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    pad_l, pad_r, pad_t, pad_b = pad_lrtb

    # Step 1: padded crop window in source coords (floats, may be negative).
    cx1 = bx1 - bw * pad_l
    cy1 = by1 - bh * pad_t
    cx2 = bx2 + bw * pad_r
    cy2 = by2 + bh * pad_b
    cw = cx2 - cx1
    ch = cy2 - cy1

    target_w, target_h = output_size
    target_aspect = target_w / target_h
    cur_aspect = cw / ch

    # Step 2: grow whichever axis is too short to hit the target aspect.
    if cur_aspect < target_aspect:
        new_w = ch * target_aspect
        mid_x = (cx1 + cx2) * 0.5
        cx1 = mid_x - new_w * 0.5
        cx2 = mid_x + new_w * 0.5
        cw = new_w
    else:
        new_h = cw / target_aspect
        if vertical_bias == "below":
            # Anchor to the box's top edge — push the extra room downward.
            cy2 = cy1 + new_h
        else:
            mid_y = (cy1 + cy2) * 0.5
            cy1 = mid_y - new_h * 0.5
            cy2 = mid_y + new_h * 0.5
        ch = new_h

    # Step 2b: a window bigger than the source can never be filled with real
    # pixels, and shifting (step 3) can't help — so scale it down, aspect
    # intact, until it fits. The crop then shows slightly less padding than
    # requested instead of grey bars. Close-up portraits hit this constantly:
    # with a face-sized pad the window ends up wider than the photo itself.
    fit = min(1.0, w / cw, h / ch)
    if fit < 1.0:
        new_w = cw * fit
        new_h = ch * fit
        mid_x = (cx1 + cx2) * 0.5
        cx1 = mid_x - new_w * 0.5
        cx2 = mid_x + new_w * 0.5
        if vertical_bias == "below":
            cy2 = cy1 + new_h      # keep the top anchor (face stays up top)
        else:
            mid_y = (cy1 + cy2) * 0.5
            cy1 = mid_y - new_h * 0.5
            cy2 = mid_y + new_h * 0.5
        cw, ch = new_w, new_h

    # Step 3: shift the window inward when it falls off an edge — instead
    # of letterboxing. "Top out → take from below" and friends. Only when
    # the window is larger than the source in a dimension does any grey
    # actually remain, and only in that dimension.
    if cx1 < 0:
        cx2 -= cx1   # cx1 is negative, this nudges cx2 right by |cx1|
        cx1 = 0
    if cx2 > w:
        cx1 -= (cx2 - w)
        cx2 = w
    if cy1 < 0:
        cy2 -= cy1
        cy1 = 0
    if cy2 > h:
        cy1 -= (cy2 - h)
        cy2 = h

    # After shifting, source bounds may still escape if the source frame
    # is genuinely smaller than the requested window. Clip and let the
    # remaining gap be filled with grey on the canvas.
    sx1 = max(0, int(round(cx1)))
    sy1 = max(0, int(round(cy1)))
    sx2 = min(w, int(round(cx2)))
    sy2 = min(h, int(round(cy2)))
    if sx2 <= sx1 or sy2 <= sy1:
        return None

    src_w = sx2 - sx1
    src_h = sy2 - sy1
    canvas_w = int(round(cw))
    canvas_h = int(round(ch))
    canvas = np.full((canvas_h, canvas_w, 3), pad_color, dtype=np.uint8)
    # Rounding can leave the source 1 px taller/wider than the canvas
    # (or vice versa) — take only as much as fits in BOTH so the paste
    # is shape-consistent. Centre whichever dimension has slack.
    paste_w = min(src_w, canvas_w)
    paste_h = min(src_h, canvas_h)
    dst_x = (canvas_w - paste_w) // 2
    dst_y = (canvas_h - paste_h) // 2
    canvas[dst_y:dst_y + paste_h, dst_x:dst_x + paste_w] = \
        img[sy1:sy1 + paste_h, sx1:sx1 + paste_w]

    return cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_AREA)
