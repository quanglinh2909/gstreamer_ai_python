from typing import Optional
from pydantic import BaseModel



class FaceRecognitionDTO(BaseModel):
    cameraId: str
    primaryConf: float
    secondaryConf: float
    maxFps: int = 5
    enabled: bool = True
    polygons: str
    tracker: Optional[str] = "ocsort"
    overlap_threshold: Optional[float] = 0.30
    dwellSeconds: Optional[int] = 0
    # Lưu khung phát hiện xuống DB để XEM LẠI vẽ được box/pose và tìm
    # sự kiện theo vùng vẽ trên hình. MẶC ĐỊNH TẮT: bật lên là ghi liên
    # tục mỗi khung hình, chỉ nên bật cho camera thực sự cần tra cứu lại.
    saveDetections: Optional[bool] = False
    # Co ghi su kien cua AI nay xuong DB/dia hay khong. MAC DINH BAT: bo
    # trong (client cu) phai giu nguyen hanh vi cu chu khong duoc tat ngam.
    saveEvents: Optional[bool] = True
    # Cách làm (biến thể) của loại AI này — id lấy từ GET /ai-variants/{type}.
    # Bỏ trống = dùng biến thể mặc định; loại chỉ có một cách làm thì luôn
    # bỏ trống vì giao diện không hiện ô chọn.
    variant: Optional[str] = None
