# -*- coding: utf-8 -*-
import asyncio
import time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app.core.config import settings
from app.services.process_ai_service import process_ai_service
from app.utils.ai_debug_overlay import draw_overlay

router = APIRouter()
prefix = "/ai-debug"
tags = ["AI Debug"]

# How often to re-arm the C++ engine's debug encoding while a client watches.
# Must be comfortably under the engine's 5 s arm TTL so the stream never
# stutters; when the viewer disconnects we stop arming and the engine stops
# encoding debug frames within a few seconds — no stuck-on CPU.
_ARM_INTERVAL_S = 2.0


async def _arm_debug(job_id: str) -> None:
    """Best-effort: ask the C++ AI engine to keep encoding live frames for
    this job even when it has no detections, so the debug view is continuous
    instead of frozen between objects. Fire-and-forget — if the engine is
    momentarily unreachable the stream just shows no new frame for a tick."""
    try:
        async with httpx.AsyncClient(
            base_url=settings.AI_API_BASE_URL, timeout=2.0,
        ) as client:
            await client.post(f"/ai-jobs/{job_id}/debug-arm")
    except Exception:
        pass

# Poll the frame cache at most this often. AI engines run at 5–15 fps;
# polling faster wastes CPU on identical frames (we deduplicate by seq
# anyway). 40 ms ≈ 25 Hz cap.
_POLL_INTERVAL_S = 0.04
# How long to wait for the very first frame before giving up — covers
# the case where the AI job isn't running yet or the C++ engine hasn't
# pushed the first sample. After this, the stream just ends cleanly.
_INITIAL_FRAME_TIMEOUT_S = 15.0
# Decode + draw + re-encode the debug overlay at half resolution. The
# overlay only needs to be human-readable, and halving each side cuts the
# JPEG decode AND encode cost to roughly a quarter — the difference between
# a debug viewer costing ~1 CPU core and ~0.25. Detection coordinates are
# scaled to match inside draw_overlay.
_DEBUG_DECODE_SCALE = 0.5


@router.get("/cameras/{camera_id}/jobs/{job_id}/mjpeg")
async def mjpeg(camera_id: str, job_id: str, request: Request):
    """Annotated MJPEG stream for one AI job.

    Subscribes on connect so the recv loop starts stashing frames; the
    fast path inside `_stash_debug_frame` is a single dict-lookup when no
    one is subscribed, so this endpoint costs nothing while idle.
    Unsubscribes in `finally` once the loop exits — which now happens
    promptly because we poll `request.is_disconnected()` every tick."""
    process_ai_service.subscribe_debug(camera_id, job_id)

    boundary = b"--frame"

    async def gen():
        arm_tasks: set = set()

        def arm():
            t = asyncio.create_task(_arm_debug(job_id))
            arm_tasks.add(t)
            t.add_done_callback(arm_tasks.discard)

        try:
            last_seq = -1
            waited = 0.0
            last_arm = 0.0
            # Arm immediately so the engine starts emitting frames at once,
            # not only after the first 2 s tick.
            arm()
            last_arm = time.monotonic()
            while True:
                # CRITICAL: Starlette does NOT reliably close a streaming
                # generator when the browser tab is closed — a send into the
                # OS socket buffer can keep succeeding for a while, so without
                # this explicit check the loop would keep decoding + drawing +
                # re-encoding every frame forever, pegging a CPU core long
                # after the viewer is gone. Check before doing any heavy work.
                if await request.is_disconnected():
                    return
                # Keep the engine's debug encoding alive while we watch.
                if time.monotonic() - last_arm >= _ARM_INTERVAL_S:
                    arm()
                    last_arm = time.monotonic()
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
                # Use the job's own primary_conf as the overlay's display
                # filter so the drawn boxes match what the pipeline keeps.
                annotated = await asyncio.to_thread(
                    draw_overlay,
                    frame["meta"], frame["full_jpeg"], frame["polygons"],
                    frame.get("primary_conf", 0.3), _DEBUG_DECODE_SCALE,
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
