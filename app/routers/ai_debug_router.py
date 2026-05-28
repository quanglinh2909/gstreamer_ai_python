# -*- coding: utf-8 -*-
import asyncio

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse

from app.services.process_ai_service import process_ai_service
from app.utils.ai_debug_overlay import draw_overlay

router = APIRouter()
prefix = "/ai-debug"
tags = ["AI Debug"]

# Poll the frame cache at most this often. AI engines run at 5–15 fps;
# polling faster wastes CPU on identical frames (we deduplicate by seq
# anyway). 40 ms ≈ 25 Hz cap.
_POLL_INTERVAL_S = 0.04
# How long to wait for the very first frame before giving up — covers
# the case where the AI job isn't running yet or the C++ engine hasn't
# pushed the first sample. After this, the stream just ends cleanly.
_INITIAL_FRAME_TIMEOUT_S = 15.0


@router.get("/cameras/{camera_id}/jobs/{job_id}/mjpeg")
async def mjpeg(camera_id: str, job_id: str):
    """Annotated MJPEG stream for one AI job.

    Subscribes on connect so the recv loop starts stashing frames; the
    fast path inside `_stash_debug_frame` is a single dict-lookup when no
    one is subscribed, so this endpoint costs nothing while idle.
    Unsubscribes in `finally`, which the StreamingResponse triggers when
    the client disconnects."""
    process_ai_service.subscribe_debug(camera_id, job_id)

    boundary = b"--frame"

    async def gen():
        try:
            last_seq = -1
            waited = 0.0
            while True:
                frame = process_ai_service.get_latest_debug_frame(camera_id, job_id)
                if frame is None:
                    # No frame yet — bail after a grace period so a wrong
                    # cam_id / inactive job doesn't hang the connection.
                    waited += _POLL_INTERVAL_S
                    if waited >= _INITIAL_FRAME_TIMEOUT_S:
                        return
                    await asyncio.sleep(_POLL_INTERVAL_S)
                    continue
                waited = 0.0  # got at least one frame; clear timeout window

                if frame["seq"] == last_seq:
                    await asyncio.sleep(_POLL_INTERVAL_S)
                    continue
                last_seq = frame["seq"]

                # OpenCV encode is CPU-bound — punt to the threadpool so
                # we don't stall other FastAPI requests.
                annotated = await asyncio.to_thread(
                    draw_overlay,
                    frame["meta"], frame["full_jpeg"], frame["polygons"],
                )
                if not annotated:
                    # draw_overlay refuses unrenderable frames (empty
                    # payload, decode failure). Skip this tick rather
                    # than yielding a zero-byte chunk that some browsers
                    # render as a broken image.
                    await asyncio.sleep(_POLL_INTERVAL_S)
                    continue
                yield (
                    boundary + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(annotated)}\r\n\r\n".encode()
                    + annotated + b"\r\n"
                )
        finally:
            process_ai_service.unsubscribe_debug(camera_id, job_id)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )


@router.get("/cameras/{camera_id}/jobs/{job_id}/view", response_class=HTMLResponse)
async def view(camera_id: str, job_id: str):
    """Bare HTML viewer — open this URL in a browser to debug live."""
    src = f"/ai-debug/cameras/{camera_id}/jobs/{job_id}/mjpeg"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AI debug {camera_id}/{job_id}</title>
<style>body{{margin:0;background:#111;color:#ccc;font:14px monospace}}
img{{max-width:100vw;max-height:100vh;display:block;margin:auto}}</style>
</head><body>
<img src="{src}" alt="AI debug stream">
</body></html>"""
