from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ParkingLotCreate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    face_camera_id: str = Field(..., min_length=1, max_length=255)
    plate_camera_id: str = Field(..., min_length=1, max_length=255)


class ParkingLotUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    face_camera_id: str = Field(..., min_length=1, max_length=255)
    plate_camera_id: str = Field(..., min_length=1, max_length=255)


class ParkingLotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: Optional[str] = None
    face_camera_id: str
    plate_camera_id: str
