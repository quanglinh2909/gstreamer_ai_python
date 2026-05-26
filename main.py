# -*- coding: utf-8 -*-
import logging
import threading
from contextlib import asynccontextmanager

from app.core.config import settings
from app.services.process_ai_service import process_ai_service

logging.basicConfig(level=logging.INFO)

import uvicorn
from fastapi import FastAPI

from app.app import api_router
from app.core.database import Base, engine
from app.models import ai_config  # noqa: F401 - đăng ký model vào Base.metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    threading.Thread(target=process_ai_service.start, daemon=True).start()
    yield
    process_ai_service.stop()


app = FastAPI(
    docs_url="/",
    root_path="/gstreamer-ai",
    openapi_url="/openapi.json",
    title="GStreamer AI API",
    lifespan=lifespan,
)

# app.include_router(api_router_ws, prefix="/ws")

app.include_router(api_router)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
