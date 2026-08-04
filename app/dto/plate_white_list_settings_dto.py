from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PlateWhiteListSettingsUpdate(BaseModel):
    # Giới hạn ở tầng API để một giá trị gõ nhầm không âm thầm vô hiệu hoá
    # barrier (pre_time cực lớn) hay mở nhầm cho xe khác (max_edit_distance
    # cao). Xem mô tả từng trường trong models/plate_white_list_settings.py.
    pre_time: int = Field(
        0, ge=0, le=3600,
        description="Giây tối thiểu giữa 2 lần mở cổng cho cùng một biển. "
                    "0 = mỗi biển chỉ mở được đúng một lần.",
    )
    max_edit_distance: int = Field(
        0, ge=0, le=3,
        description="Số ký tự tối đa được phép sai so với biển trong "
                    "whitelist. 0 = khớp tuyệt đối. Càng cao càng dễ mở "
                    "nhầm cho xe có biển gần giống.",
    )
    ocr_confidence: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Ngưỡng tin cậy của từng ký tự OCR khi đọc biển cho "
                    "nhánh whitelist. Ký tự yếu hơn bị loại khỏi chuỗi.",
    )
    min_plate_length: int = Field(
        7, ge=1, le=12,
        description="Số ký tự tối thiểu của biển đọc được thì mới đối chiếu "
                    "whitelist.",
    )
    barrier_duration: float = Field(
        0.5, gt=0, le=10,
        description="Độ dài xung mở barrier (giây).",
    )
    gate_group_id: Optional[int] = Field(
        None,
        description="ID CỤM CỔNG mà camera thuộc về (tab Cụm cổng). Khi "
                    "thuộc cụm, thời gian chờ lấy của CỤM và 'pre_time' ở "
                    "trên KHÔNG còn tác dụng. null = camera đứng riêng.",
    )


class PlateWhiteListSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    camera_id: str
    pre_time: int
    max_edit_distance: int
    ocr_confidence: float
    min_plate_length: int
    barrier_duration: float
    gate_group_id: Optional[int] = None
