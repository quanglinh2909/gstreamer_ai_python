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
    # Co ghi su kien cua AI nay xuong DB/dia hay khong. MAC DINH BAT: bo
    # trong (client cu) phai giu nguyen hanh vi cu chu khong duoc tat ngam.
    saveEvents: Optional[bool] = True
    # Cách làm (biến thể) của loại AI này — id lấy từ GET /ai-variants/{type}.
    # Bỏ trống = dùng biến thể mặc định; loại chỉ có một cách làm thì luôn
    # bỏ trống vì giao diện không hiện ô chọn.
    variant: Optional[str] = None
