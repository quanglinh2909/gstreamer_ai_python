from typing import Dict

from pydantic import BaseModel


class AIEnabledCountResponse(BaseModel):
    # Tổng số AI đang bật trên tất cả camera.
    total: int
    # Số AI đang bật theo từng loại, ví dụ:
    # {"plate_recognition": 2, "face_recognition": 1, "restricted_area": 0}
    by_type: Dict[str, int]
