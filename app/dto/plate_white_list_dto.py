from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PlateWhiteListCreate(BaseModel):
    plate_number: str = Field(..., min_length=1, max_length=12)
    name: Optional[str] = Field(None, max_length=255)


class PlateWhiteListUpdate(BaseModel):
    plate_number: str = Field(..., min_length=1, max_length=12)
    name: Optional[str] = Field(None, max_length=255)


class PlateWhiteListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    plate_number: str
    name: Optional[str] = None
