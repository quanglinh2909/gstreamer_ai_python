# -*- coding: utf-8 -*-
"""Manual test endpoints for the parking-lot correlation worker. They push the
exact same task shapes that face_recognition_service / plate_recognition_service
emit, so you can exercise the face<->plate matching without a live camera."""
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.tasks.task_parking_lot import task_parking_lot
from app.utils.plate_recognition_hepper import DEFAULT_OCR_LABELS

router = APIRouter()
prefix = "/parking-lot-test"
tags = ["Parking Lot Test"]

_LABEL_TO_CLASS_ID = {label: index for index, label in enumerate(DEFAULT_OCR_LABELS)}


def _fake_ocr_children(plate: str):
    """Dựng các box OCR giả từ một chuỗi biển.

    TaskParkingLot tự đọc lại biển từ children bằng ocr_confidence của bãi,
    nên endpoint test phải cấp children chứ không cấp chuỗi sẵn — có vậy nó
    mới đi đúng đường mà camera thật đi. Các box xếp một hàng ngang, cách đều,
    score = 1.0 để mọi ngưỡng ocr_confidence đều cho qua.

    Ký tự không nằm trong bảng nhãn OCR (dấu gạch, khoảng trắng) bị bỏ — vô
    hại vì mọi bước so khớp phía sau đều chuẩn hoá về chữ + số.
    """
    children = []
    for char in plate.upper():
        class_id = _LABEL_TO_CLASS_ID.get(char)
        if class_id is None:
            continue
        left = len(children) * 20
        children.append({
            "x1": left, "y1": 0, "x2": left + 16, "y2": 34,
            "score": 1.0, "classId": class_id,
        })
    return children


class FaceTaskRequest(BaseModel):
    identity_id: int
    camera_id: str = Field(..., min_length=1)


class PlateTaskRequest(BaseModel):
    plate: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)


@router.post("/face")
async def send_face_task(payload: FaceTaskRequest):
    task = {
        "task": "face_recognition",
        "identity_id": payload.identity_id,
        "timestamp": time.time(),
        "camera_id": payload.camera_id,
    }
    task_parking_lot.add_task(task)
    return {"queued": True, "task": task}


@router.post("/plate")
async def send_plate_task(payload: PlateTaskRequest):
    children = _fake_ocr_children(payload.plate)
    task = {
        "task": "plate_recognition",
        "children": children,
        "timestamp": time.time(),
        "camera_id": payload.camera_id,
    }
    task_parking_lot.add_task(task)
    # Trả về số box thay vì cả list cho gọn; list chỉ là dữ liệu dựng máy móc.
    return {"queued": True, "plate": payload.plate, "ocr_boxes": len(children)}
