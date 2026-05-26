# -*- coding: utf-8 -*-
import logging
from contextlib import asynccontextmanager

from app.core.config import settings

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
    yield


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
