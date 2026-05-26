from typing import Optional

from pydantic import BaseModel, ConfigDict


class EventPlateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    plate_number: str
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class EventFaceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    identity_id: Optional[int] = None
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None
