from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class _WithBox(BaseModel):
    """Gộp 4 cột box_x1..box_y2 (chuẩn hoá [0,1]) thành một object `box` cho
    frontend vẽ khung; cột thô ẩn khỏi JSON (exclude) để đầu ra gọn."""

    box_x1: Optional[float] = Field(default=None, exclude=True)
    box_y1: Optional[float] = Field(default=None, exclude=True)
    box_x2: Optional[float] = Field(default=None, exclude=True)
    box_y2: Optional[float] = Field(default=None, exclude=True)

    @computed_field
    @property
    def box(self) -> Optional[dict]:
        if self.box_x1 is None or self.box_x2 is None:
            return None
        return {
            "x1": self.box_x1,
            "y1": self.box_y1,
            "x2": self.box_x2,
            "y2": self.box_y2,
        }


class EventPlateResponse(_WithBox):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    plate_number: str
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class EventFaceResponse(_WithBox):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    identity_id: Optional[int] = None
    name: Optional[str] = None
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class RestrictedAreaResponse(_WithBox):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None
