from typing import Optional

from pydantic import BaseModel


class PlateRecognitionDTO(BaseModel):
    cameraId: str
    primaryConf: float
    secondaryConf: float
    maxFps: int = 5
    enabled: bool = True
    polygons: str
    tracker: Optional[str] = "bytetrack"
    overlap_threshold: Optional[float] = 0.30
    dwellSeconds: Optional[int] = 0


