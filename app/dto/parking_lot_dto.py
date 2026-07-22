from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ParkingLotSettings(BaseModel):
    """Ngưỡng hoạt động của một cổng — xem mô tả từng cột trong
    models/parking_lot.py. Giới hạn ở đây khớp với server_default để một giá
    trị gõ nhầm không âm thầm làm cổng ngừng hoạt động."""

    time_expired: int = Field(
        30, ge=1, le=600,
        description="Cửa sổ ghép cặp mặt ↔ biển (giây).",
    )
    match_cooldown: int = Field(
        30, ge=0, le=600,
        description="Một biển chỉ tạo một sự kiện trong ngần này giây. "
                    "Chặn 2 người ngồi cùng xe tạo 2 dòng.",
    )
    barrier_duration: float = Field(
        0.5, gt=0, le=10,
        description="Độ dài xung mở barrier (giây).",
    )
    max_edit_distance: int = Field(
        2, ge=0, le=3,
        description="Số ký tự tối đa được phép sai so với biển đã đăng ký. "
                    "0 = khớp tuyệt đối.",
    )
    ocr_confidence: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Ngưỡng tin cậy của từng ký tự OCR khi bãi đọc lại biển. "
                    "Ký tự yếu hơn bị loại khỏi chuỗi.",
    )


class ParkingLotCreate(ParkingLotSettings):
    name: Optional[str] = Field(None, max_length=255)
    face_camera_id: str = Field(..., min_length=1, max_length=255)
    plate_camera_id: str = Field(..., min_length=1, max_length=255)


class ParkingLotUpdate(ParkingLotSettings):
    name: Optional[str] = Field(None, max_length=255)
    face_camera_id: str = Field(..., min_length=1, max_length=255)
    plate_camera_id: str = Field(..., min_length=1, max_length=255)


class ParkingLotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: Optional[str] = None
    face_camera_id: str
    plate_camera_id: str
    time_expired: int
    match_cooldown: int
    barrier_duration: float
    max_edit_distance: int
    ocr_confidence: float
