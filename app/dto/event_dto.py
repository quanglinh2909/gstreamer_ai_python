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


class AiEventResponse(_WithBox):
    """Các trường mà sự kiện AI nào cũng trả về — đối xứng với AiEventMixin
    bên models. Lớp con chỉ khai thêm cái riêng của mình."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: str
    confidence: float
    timestamp: int
    image_full: Optional[str] = None
    image_crop: Optional[str] = None


class EventPlateResponse(AiEventResponse):
    plate_number: str


class EventFaceResponse(AiEventResponse):
    identity_id: Optional[int] = None
    name: Optional[str] = None


class RestrictedAreaResponse(AiEventResponse):
    pass


class EventMaskResponse(AiEventResponse):
    # "wearing_mask" | "not_wearing_mask" | "unknown"
    mask_status: str
    track_id: Optional[int] = None
