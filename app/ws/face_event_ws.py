"""Fan-out face events to any number of WebSocket subscribers.

Thin wrapper over the shared CameraFilteredBroadcaster — a separate instance so
a UI can subscribe to only the face stream, optionally scoped to one camera.
See app/ws/base_event_ws.py for the threading model and per-camera filtering.
"""

from __future__ import annotations

from app.ws.base_event_ws import CameraFilteredBroadcaster

face_event_broadcaster = CameraFilteredBroadcaster("face")
