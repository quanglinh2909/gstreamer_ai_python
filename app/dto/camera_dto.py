from typing import Optional

from pydantic import BaseModel


# Backend Python chỉ chuyển tiếp sang engine C++ (HTTPXClient), nhưng
# `model_dump(exclude_none=True)` chỉ giữ những trường KHAI BÁO Ở ĐÂY — trường
# nào thiếu trong DTO là bị lặng lẽ vứt, API vẫn trả 200 mà engine không nhận
# được gì. Thêm trường mới ở engine thì phải thêm cả ở đây.
#
# motionSensitivity/motionThreshold là SỐ THỰC 0..1 (engine: Float64, thẳng vào
# thuộc tính cùng tên của motioncells). Trước đây khai `int` với mặc định
# 50/5000 — vượt xa miền cho phép nên GStreamer kẹp về 1.0, tức luôn nhạy tối đa.


class CameraCreateDTO(BaseModel):
    name: str
    rtsp: str
    hardware: str
    recordingEnabled: bool = False
    recordingMode: str = "off"
    motionEnabled: bool = False
    motionSensitivity: float = 0.5
    motionThreshold: float = 0.01
    preMotionSeconds: int = 5
    postMotionSeconds: int = 5
    segmentSeconds: int = 60
    # Lưới phát hiện chuyển động theo ô (motioncells chỉ nhận 8..32).
    motionGridX: int = 32
    motionGridY: int = 32
    motionCellLevels: str = ""
    # Vùng chuyển động, JSON [{"r1","c1","r2","c2","level"}]. Thay cho
    # motionCellLevels (một chữ số mỗi ô) — xem ghi chú ở RecordingTypes.hpp.
    motionZones: str = ""
    # Ghi sự kiện xuống DB hay chỉ bắn WebSocket (giống nhận diện khẩu trang).
    motionSaveEvents: bool = True
    # Hạn lưu theo NGÀY của riêng camera này; 0 = không giới hạn. Engine chỉ
    # cất hộ vào cột cameras.retention_days — chính bộ dọn của Python
    # (storage_cleanup_service) mới là chỗ thi hành.
    retentionDays: int = 0


class CameraUpdateDTO(BaseModel):
    name: Optional[str] = None
    rtsp: Optional[str] = None
    hardware: Optional[str] = None
    recordingEnabled: Optional[bool] = None
    recordingMode: Optional[str] = None
    motionEnabled: Optional[bool] = None
    motionSensitivity: Optional[float] = None
    motionThreshold: Optional[float] = None
    preMotionSeconds: Optional[int] = None
    postMotionSeconds: Optional[int] = None
    segmentSeconds: Optional[int] = None
    motionGridX: Optional[int] = None
    motionGridY: Optional[int] = None
    motionCellLevels: Optional[str] = None
    motionZones: Optional[str] = None
    motionSaveEvents: Optional[bool] = None
    retentionDays: Optional[int] = None
