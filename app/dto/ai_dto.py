from typing import Dict

from pydantic import BaseModel


class AIEnabledCountResponse(BaseModel):
    # Tổng số AI đang bật trên tất cả camera.
    total: int
    # Số AI đang bật theo từng loại, ví dụ:
    # {"plate_recognition": 2, "face_recognition": 1, "restricted_area": 0}
    by_type: Dict[str, int]
    # Số CAMERA đang bật phát hiện chuyển động.
    #
    # Để RIÊNG chứ không nhét vào by_type: chuyển động không phải một "AI job"
    # — nó không chạy model nào, không dùng NPU, và đếm theo CAMERA chứ không
    # theo job. Cộng vào `total` là con số "AI đang bật" hết nghĩa.
    motion_cameras: int = 0
    # Trong số đó, bao nhiêu camera đang ghi hình theo chuyển động.
    motion_recording_cameras: int = 0
