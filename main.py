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
from app.core.database import Base, engine
from app.core.milvus import close_client, get_client
from app.models import ai_config, event_face, event_plate, identity  # noqa: F401 - đăng ký model vào Base.metadata

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    get_client()
    threading.Thread(target=process_ai_service.start, daemon=True).start()
    yield
    process_ai_service.stop()
    close_client()


app = FastAPI(
    docs_url="/",
    root_path="/gstreamer-ai",
    openapi_url="/openapi.json",
    title="GStreamer AI API",
    lifespan=lifespan,
)

app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# app.include_router(api_router_ws, prefix="/ws")

app.include_router(api_router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
