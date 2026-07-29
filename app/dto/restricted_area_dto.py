from typing import Optional

from pydantic import BaseModel


class RestrictedAreaDTO(BaseModel):
    cameraId: str
    primaryConf: float
    secondaryConf: Optional[float] = 0
    maxFps: int = 5
    enabled: bool = True
    polygons: str
    tracker: Optional[str] = "ocsort"
    overlap_threshold: Optional[float] = 0.30
    dwellSeconds: Optional[int] = 0
    # Chọn được từ giao diện (bỏ trống = dùng mặc định của RESTRICTED_AREA_SPEC).
    # modelFile: tên file .rknn trong danh sách /ai-models của engine.
    # modelType: một trong các loại engine hỗ trợ (GET /ai-model-types →
    #   yolov8_detect, yolov8_pose, yolov8_seg, rf_detect, face_recognition) —
    #   phải khớp kiến trúc của model. Nhận thẳng, không giới hạn cứng ở đây.
    # classFilter: CSV id lớp giữ lại, vd "0" (person) hay "0,1,2"; "" = giữ tất cả.
    modelFile: Optional[str] = None
    modelType: Optional[str] = None
    classFilter: Optional[str] = None
    # Lưu khung phát hiện xuống DB để XEM LẠI vẽ được box/pose và tìm
    # sự kiện theo vùng vẽ trên hình. MẶC ĐỊNH TẮT: bật lên là ghi liên
    # tục mỗi khung hình, chỉ nên bật cho camera thực sự cần tra cứu lại.
    saveDetections: Optional[bool] = False
