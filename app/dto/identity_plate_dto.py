from pydantic import BaseModel, ConfigDict, Field


class IdentityPlateCreate(BaseModel):
    plate_number: str = Field(..., min_length=1, max_length=12)


class IdentityPlateUpdate(BaseModel):
    plate_number: str = Field(..., min_length=1, max_length=12)


class IdentityPlateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    identity_id: int
    plate_number: str
