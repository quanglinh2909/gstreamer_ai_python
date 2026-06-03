# -*- coding: utf-8 -*-
"""Manual test endpoints for the parking-lot correlation worker. They push the
exact same task shapes that face_recognition_service / plate_recognition_service
emit, so you can exercise the face<->plate matching without a live camera."""
import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.tasks.task_parking_lot import task_parking_lot

router = APIRouter()
prefix = "/parking-lot-test"
tags = ["Parking Lot Test"]


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
    task = {
        "task": "plate_recognition",
        "plate": payload.plate,
        "timestamp": time.time(),
        "camera_id": payload.camera_id,
    }
    task_parking_lot.add_task(task)
    return {"queued": True, "task": task}
