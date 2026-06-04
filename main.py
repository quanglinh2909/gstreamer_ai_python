# -*- coding: utf-8 -*-
import logging
import threading
from contextlib import asynccontextmanager

from app.core.config import settings
from app.services.process_ai_service import process_ai_service

logging.basicConfig(level=logging.INFO)

import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.app import api_router
from app.core.database import AsyncSessionLocal, Base, engine
from app.core.milvus import close_client, get_client
from app.repositories.face_vector_repository import FaceVectorRepository
from app.models import ai_config, event_face, event_plate, identity, identity_plate, parking_lot, parking_lot_event, plate_white_list, restricted_areas, system_metrics  # noqa: F401 - đăng ký model vào Base.metadata
from app.services.identity_plate_service import identity_plate_service
from app.services.parking_lot_service import parking_lot_service
from app.services.plate_white_list_service import plate_white_list_service
from app.tasks.task_parking_lot import task_parking_lot
from app.tasks.task_system_metrics import task_system_metrics

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Prime the plate whitelist cache before the ALPR consumer thread
    # starts so the very first detection can hit the in-memory map.
    async with AsyncSessionLocal() as db:
        await plate_white_list_service.load_all(db)
        await parking_lot_service.load_all(db)
        await identity_plate_service.load_all(db)
    get_client()
    # Mirror the Milvus face vectors into RAM so per-frame matching is an
    # in-memory cosine search instead of a gRPC round-trip.
    FaceVectorRepository.load_all_to_cache()
    threading.Thread(target=process_ai_service.start, daemon=True).start()
    # Drains the face/plate task queue and correlates them per parking lot.
    threading.Thread(target=task_parking_lot.worker, daemon=True).start()
    # Samples CPU/temp/memory/load/NPU/RGA every 10s into rolling 1-month tables.
    threading.Thread(target=task_system_metrics.worker, daemon=True).start()
    yield
    process_ai_service.stop()
    task_system_metrics.stop()
    close_client()


app = FastAPI(
    docs_url="/",
    openapi_url="/openapi.json",
    title="GStreamer AI API",
    lifespan=lifespan,
)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# app.include_router(api_router_ws, prefix="/ws")

app.include_router(api_router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
