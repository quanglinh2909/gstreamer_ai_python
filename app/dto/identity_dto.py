from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class IdentityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    mac_bluetooth: Optional[str] = None
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class FaceInfo(BaseModel):
    id: int
    score: float
    embedding: List[float]
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class IdentityWithFaceResponse(BaseModel):
    id: int
    name: str
    mac_bluetooth: Optional[str] = None
    image_full: Optional[str] = None
    image_crop: Optional[str] = None
    face: Optional[FaceInfo] = None
