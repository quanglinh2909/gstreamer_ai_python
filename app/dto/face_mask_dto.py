from typing import Optional

from pydantic import BaseModel


class FaceMaskDTO(BaseModel):
    cameraId: str
    primaryConf: float
    secondaryConf: Optional[float] = 0
    maxFps: int = 5
    enabled: bool = True
    polygons: str
    tracker: Optional[str] = "ocsort"
    overlap_threshold: Optional[float] = 0.30
    dwellSeconds: Optional[int] = 0
    # How many consecutive confirmations before firing the mask/face event.
    # Was hardcoded to 3 in the service; now driven per-config via extra_data.
    count_confirm: Optional[int] = 3
    # Re-alert interval in seconds while the same person stays in the zone.
    # 0 disables re-alerting (fire once per presence).
    re_alert_seconds: Optional[int] = 0
    # Độ dài xung mở barrier (giây) khi phát hiện người KHÔNG đeo khẩu trang.
    # Tuỳ phần cứng từng cổng; trước fix cứng 0.5 trong face_mask_service.
    barrier_duration: Optional[float] = 0.5
