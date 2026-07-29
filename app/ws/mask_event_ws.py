"""Fan-out face-mask (access-control) events to WebSocket subscribers.

Thin wrapper over the shared CameraFilteredBroadcaster — a separate instance so
a UI can subscribe to only the mask stream, optionally scoped to one camera.
Unlike the others, mask events are NOT persisted to a table; the crop image
travels inline as a base64 data URL on the frame (see
push_envent_metadata.push_event), so the payload is self-contained and no
/uploads file is needed. See app/ws/base_event_ws.py for the threading model
and per-camera filtering.
"""

from __future__ import annotations

from app.ws.base_event_ws import CameraFilteredBroadcaster

mask_event_broadcaster = CameraFilteredBroadcaster("mask")
