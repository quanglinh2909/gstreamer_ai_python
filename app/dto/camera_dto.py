from typing import Optional

from pydantic import BaseModel


class CameraCreateDTO(BaseModel):
    name: str
    rtsp: str
    hardware: str
    recordingEnabled: bool = False
    recordingMode: str = "off"
    motionEnabled: bool = False
    motionSensitivity: int = 50
    motionThreshold: int = 5000
    preMotionSeconds: int = 5
    postMotionSeconds: int = 5
    segmentSeconds: int = 60
    motionKeyframeOnly: bool = False


class CameraUpdateDTO(BaseModel):
    name: Optional[str] = None
    rtsp: Optional[str] = None
    hardware: Optional[str] = None
    recordingEnabled: Optional[bool] = None
    recordingMode: Optional[str] = None
    motionEnabled: Optional[bool] = None
    motionSensitivity: Optional[int] = None
    motionThreshold: Optional[int] = None
    preMotionSeconds: Optional[int] = None
    postMotionSeconds: Optional[int] = None
    segmentSeconds: Optional[int] = None
    motionKeyframeOnly: Optional[bool] = None
