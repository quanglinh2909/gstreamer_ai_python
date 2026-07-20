from typing import Optional

from pydantic import BaseModel


class PlateRecognitionDTO(BaseModel):
    cameraId: str
    primaryConf: float
    secondaryConf: float
    maxFps: int = 5
    enabled: bool = True
    polygons: str
    tracker: Optional[str] = "ocsort"
    overlap_threshold: Optional[float] = 0.30
    dwellSeconds: Optional[int] = 0
    # Giây tối thiểu giữa hai lần mở barrier cho cùng một biển số (chống mở
    # lặp). Trước fix cứng = 10 trong plate_white_list_service; giờ lưu vào
    # extra_data để cấu hình được.
    pre_time: Optional[int] = 10


