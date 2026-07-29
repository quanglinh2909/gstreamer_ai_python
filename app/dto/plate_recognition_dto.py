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
    # Số ký tự tối thiểu của chuỗi biển thì mới ghi một dòng EventPlate.
    # Ngưỡng RIÊNG của nhánh lưu sự kiện — nhánh mở barrier có
    # PlateWhiteListSettings.min_plate_length của nó, thường dễ hơn.
    #
    # Không có giá trị mặc định ở bất kỳ đâu trong backend: để None nghĩa là
    # AI job chưa cấu hình ngưỡng, và khi đó camera KHÔNG ghi EventPlate cho
    # tới khi người dùng đặt giá trị trong Cấu hình AI.
    min_plate_length: Optional[int] = None
    # pre_time đã bỏ khỏi đây: các ngưỡng của nhánh whitelist/barrier
    # (pre_time, max_edit_distance, ocr_confidence, min_plate_length)
    # nay nằm trong bảng plate_white_list_settings theo từng camera, cấu hình
    # qua API /plate-white-list-settings/{camera_id}. Client cũ vẫn gửi kèm
    # pre_time thì pydantic bỏ qua, không lỗi.
    # Lưu khung phát hiện xuống DB để XEM LẠI vẽ được box/pose và tìm
    # sự kiện theo vùng vẽ trên hình. MẶC ĐỊNH TẮT: bật lên là ghi liên
    # tục mỗi khung hình, chỉ nên bật cho camera thực sự cần tra cứu lại.
    saveDetections: Optional[bool] = False
