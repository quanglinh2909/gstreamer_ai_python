from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class IdentityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class FaceInfo(BaseModel):
    id: int
    score: float
    embedding: List[float]


class IdentityWithFaceResponse(BaseModel):
    id: int
    name: str
    face: Optional[FaceInfo] = None
