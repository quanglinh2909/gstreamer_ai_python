# -*- coding: utf-8 -*-

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.utils.open_door.door_manager import door_manager

router = APIRouter()
prefix = "/barrier"
tags = ["Barrier"]

# Pins wired to a barrier relay in GPIOBarrieOrangePi.open_barrie. Any other pin
# is a no-op in the GPIO layer, so reject it up front with a clear error.
ALLOWED_PINS = {3, 5, 19, 21}


class OpenBarrierRequest(BaseModel):
    io_pin: int = Field(..., description="Relay pin to pulse (one of 3, 5, 19, 21)")


@router.post("/open")
async def open_barrier(payload: OpenBarrierRequest):
    if payload.io_pin not in ALLOWED_PINS:
        raise HTTPException(
            status_code=400,
            detail=f"io_pin must be one of {sorted(ALLOWED_PINS)}",
        )
    try:
        # open_barrie spins up a short-lived daemon thread per pin and returns
        # immediately, but keep it off the event loop in case GPIO.setup blocks.
        await asyncio.to_thread(door_manager.open_door, payload.io_pin)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open barrier: {e}")
    return {"opened": True, "io_pin": payload.io_pin}
