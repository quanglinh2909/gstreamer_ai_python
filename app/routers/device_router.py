from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import os
import time
import asyncio
import json
from app.utils.push_envent_metadata import push_event_metadata

router = APIRouter()


@router.get("/device/info")
def get_device_info(request: Request):
    """Get device metadata information from environment variables."""
    try:
        device_info = {
            "device_type": os.environ.get("DEVICE_TYPE", "ACCESS_CONTROL"),
            "device_class": os.environ.get("DEVICE_CLASS", "OAC"),
            "serial_number": os.environ.get("SERIAL_NUMBER", "ACCESS-CONTROL-001"),
            "hardware_version": os.environ.get("HARDWARE_VERSION", "v1.0"),
            "software_version": os.environ.get("SOFTWARE_VERSION", "v1.0"),
            "vendor": os.environ.get("VENDOR", "Oryza"),
            "machine_name": os.environ.get("MACHINE_NAME", "Box-AI-001"),
        }

        return JSONResponse(content=device_info, status_code=200)

    except Exception as e:
        # logger.error(f"Error getting device info: {e}")
        return JSONResponse(
            content={"success": False, "message": "Failed to get device info"},
            status_code=500,
        )


@router.get("/ai/capabilities")
def get_ai_capabilities(request: Request):
    """Get AI capabilities information from environment variables."""
    try:
        # Parse supported AI types from environment variable
        supported_ai_types_str = os.environ.get(
            "SUPPORTED_AI_TYPES",
            "AccessControl",
        )
        supported_ai_types = [
            ai_type.strip() for ai_type in supported_ai_types_str.split(",")
        ]

        ai_capabilities = {
            "supported_ai_types": supported_ai_types,
        }

        return JSONResponse(content=ai_capabilities, status_code=200)

    except Exception as e:
        # logger.error(f"Error getting AI capabilities: {e}")
        return JSONResponse(
            content={"success": False, "message": "Failed to get AI capabilities"},
            status_code=500,
        )


@router.get("/events/stream")
async def stream_events(request: Request):
    """
    Stream events in real-time using multipart boundary streaming.
    
    Returns multipart stream with:
    - Heartbeat every 3 seconds (text/plain)
    - JSON metadata with bbox coordinates + person analysis (application/json)
    - Full JPEG image (image/jpeg, X-Image-Type: full)
    - Cropped JPEG image from bbox (image/jpeg, X-Image-Type: cropped) - if available
    
    Person analysis includes: age, gender, clothing colors, accessories, etc.
    """

    async def event_stream_generator():
        """Generator function for multipart event streaming"""
        boundary = "myboundary"

        # Heartbeat mechanism
        last_heartbeat_time = time.time()
        heartbeat_interval = 3.0  # 3 seconds

        print("📡 Client connected to /events/stream")

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    print("❌ Client disconnected from /events/stream")
                    break

                current_time = time.time()

                # Send heartbeat every 3 seconds
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    heartbeat_text = f"heartbeat {int(current_time)}"

                    yield f"--{boundary}\r\n".encode("utf-8")
                    yield "Content-Type: text/plain\r\n".encode("utf-8")
                    yield f"Content-Length: {len(heartbeat_text.encode('utf-8'))}\r\n\r\n".encode("utf-8")
                    yield heartbeat_text.encode("utf-8")
                    yield f"\r\n--{boundary}--\r\n".encode("utf-8")

                    last_heartbeat_time = current_time
                    print(f"💓 Heartbeat sent at {int(current_time)}")

                    # Process events from queue
                    if push_event_metadata.event_queue:
                        event_data = push_event_metadata.event_queue.pop(0)
                        print(f"📤 Streaming event: {event_data['id']}")

                        # Lấy kết quả phân tích đã được tính sẵn từ /alerts
                        analysis_result = event_data.get("analysis")
                        if analysis_result:
                            print(f"🔍 Using pre-analyzed result: {analysis_result}")

                        # 1. Send JSON metadata (including analysis)
                        json_data = {
                            "id": event_data["id"],
                            "type": event_data["type"],
                            "timestamp": event_data["timestamp"],
                            "mask_status": event_data["mask_status"],
                            "detection_class": event_data["detection_class"],
                            "direction": event_data["direction"],
                            "velocity_x": event_data["velocity_x"],
                            "velocity_y": event_data["velocity_y"],
                            "bbox_x1": event_data.get("bbox_x1", 0),
                            "bbox_y1": event_data.get("bbox_y1", 0),
                            "bbox_x2": event_data.get("bbox_x2", 0),
                            "bbox_y2": event_data.get("bbox_y2", 0),
                            "has_cropped_image": event_data.get("cropped_bytes") is not None,
                            "analysis": analysis_result  # Add analysis result
                        }
                        json_text = json.dumps(json_data)

                        yield f"--{boundary}\r\n".encode("utf-8")
                        yield "Content-Type: application/json\r\n".encode("utf-8")
                        yield f"Content-Length: {len(json_text.encode('utf-8'))}\r\n\r\n".encode("utf-8")
                        yield json_text.encode("utf-8")
                        yield f"\r\n--{boundary}\r\n".encode("utf-8")

                        print(f"✅ JSON sent: {len(json_text)} bytes")

                        # 2. Send full JPEG image
                        image_bytes = event_data["image_bytes"]

                        yield f"--{boundary}\r\n".encode("utf-8")
                        yield "Content-Type: image/jpeg\r\n".encode("utf-8")
                        yield "X-Image-Type: full\r\n".encode("utf-8")
                        yield f"Content-Length: {len(image_bytes)}\r\n\r\n".encode("utf-8")
                        yield image_bytes
                        yield f"\r\n--{boundary}\r\n".encode("utf-8")

                        print(f"✅ Full image sent: {len(image_bytes)} bytes")

                        # 3. Send cropped JPEG image (if available)
                        if event_data.get("cropped_bytes"):
                            cropped_bytes = event_data["cropped_bytes"]

                            yield f"--{boundary}\r\n".encode("utf-8")
                            yield "Content-Type: image/jpeg\r\n".encode("utf-8")
                            yield "X-Image-Type: cropped\r\n".encode("utf-8")
                            yield f"Content-Length: {len(cropped_bytes)}\r\n\r\n".encode("utf-8")
                            yield cropped_bytes
                            yield f"\r\n--{boundary}--\r\n".encode("utf-8")

                            print(f"✅ Cropped image sent: {len(cropped_bytes)} bytes")
                        else:
                            print("⚠️  No cropped image available")

                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"❌ Error in event stream: {e}")
            # Send error response
            error_text = json.dumps({
                "type": "error",
                "message": str(e),
                "timestamp": time.time()
            })
            yield f"--{boundary}\r\n".encode("utf-8")
            yield "Content-Type: application/json\r\n".encode("utf-8")
            yield f"Content-Length: {len(error_text.encode('utf-8'))}\r\n\r\n".encode("utf-8")
            yield error_text.encode("utf-8")
            yield f"\r\n--{boundary}--\r\n".encode("utf-8")
        finally:
            print("🔌 Stream ended")

    response = StreamingResponse(
        event_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=myboundary",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
    return response
