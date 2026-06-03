from typing import Optional

from pydantic import BaseModel, ConfigDict


class ParkingLotEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    parking_lot_id: Optional[int] = None
    parking_lot_name: Optional[str] = None
    identity_id: Optional[int] = None
    name: Optional[str] = None  # identity name
    plate_number: str
    face_camera_id: Optional[str] = None
    plate_camera_id: Optional[str] = None
    face_image_full: Optional[str] = None
    plate_image_full: Optional[str] = None
    timestamp: int
