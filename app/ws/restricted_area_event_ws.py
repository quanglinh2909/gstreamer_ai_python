"""Fan-out restricted-area events to any number of WebSocket subscribers.

Thin wrapper over the shared CameraFilteredBroadcaster — a separate instance so
a UI can subscribe to only the restricted-area stream, optionally scoped to one
camera. See app/ws/base_event_ws.py for the threading model and per-camera
filtering.
"""

from __future__ import annotations

from app.ws.base_event_ws import CameraFilteredBroadcaster

restricted_area_event_broadcaster = CameraFilteredBroadcaster("restricted-area")
