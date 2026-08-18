# -*- coding: utf-8 -*-
import logging
import threading
from contextlib import asynccontextmanager

from app.core.config import settings
from app.services.process_ai_service import process_ai_service
from app.utils.play_sound import play_sound

logging.basicConfig(level=logging.INFO)

import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.app import api_router
from app.core.database import AsyncSessionLocal, Base, engine
from app.core.milvus import close_client, get_client
from app.repositories.face_vector_repository import FaceVectorRepository
from app.models import ai_config, detection_slice, event_face, event_mask, event_plate, identity, identity_plate, parking_lot, parking_lot_event, plate_gate_group, plate_white_list, plate_white_list_settings, restricted_areas, storage_policy, system_metrics  # noqa: F401 - đăng ký model vào Base.metadata
from app.services.identity_plate_service import identity_plate_service
from app.services.parking_lot_service import parking_lot_service
from app.services.plate_white_list_service import plate_white_list_service
from app.services.plate_white_list_settings_service import plate_white_list_settings_service
from app.moq.server import moq_server
from app.tasks.task_parking_lot import task_parking_lot
from app.tasks.task_storage_cleanup import task_storage_cleanup
from app.tasks.task_system_metrics import task_system_metrics

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Seed hàng cấu hình dọn-dung-lượng mặc định (bảng một-hàng, id=1).
        await conn.exec_driver_sql(
            "INSERT INTO storage_policy (id) VALUES (1) ON CONFLICT (id) DO NOTHING"
        )
        # create_all KHÔNG thêm cột vào bảng đã tồn tại. Cột cấu hình mới phải
        # tự ALTER (idempotent) để bãi đang chạy nhận được mà không mất dữ liệu.
        await conn.exec_driver_sql(
            "ALTER TABLE parking_lot "
            "ADD COLUMN IF NOT EXISTS face_confidence DOUBLE PRECISION "
            "NOT NULL DEFAULT 0.15"
        )
        # Hai loại dữ liệu mới vào danh sách tự dọn (khẩu trang + chuyển động).
        # Bãi đang chạy đã có hàng id=1 nên INSERT ở trên không đụng tới, phải
        # ALTER thì cột mới xuất hiện với đúng giá trị mặc định.
        for column in ("w_event_mask", "w_motion_event"):
            await conn.exec_driver_sql(
                f"ALTER TABLE storage_policy "
                f"ADD COLUMN IF NOT EXISTS {column} DOUBLE PRECISION "
                f"NOT NULL DEFAULT 4"
            )
        # Cụm cổng của nhánh whitelist/barrier. Bảng plate_gate_group do
        # create_all ở trên tạo, nhưng create_all KHÔNG thêm cột vào bảng đã
        # tồn tại nên cột trỏ tới cụm phải tự ALTER.
        #
        # DROP cột `gate_group` (VARCHAR) của bản trước: khi ấy cụm chỉ là một
        # cái nhãn gõ tay, nên cụm gồm camera chờ 30s và camera chờ 20s thì
        # không có câu trả lời đúng nào cho "chờ bao lâu". Giờ cụm là một bảng
        # có thời gian chờ của chính nó.
        await conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS plate_white_list_settings "
            "DROP COLUMN IF EXISTS gate_group"
        )
        await conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS plate_white_list_settings "
            "ADD COLUMN IF NOT EXISTS gate_group_id INTEGER"
        )
        # Ảnh khung hình của sự kiện chuyển động. Bảng motion_events do engine
        # C++ tạo (sql/init.sql) nên create_all ở trên không biết
        # tới nó — cột phải tự thêm ở đây, và phải chịu được lúc engine chưa
        # từng chạy trên máy này.
        await conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS motion_events "
            "ADD COLUMN IF NOT EXISTS image_path TEXT NOT NULL DEFAULT ''"
        )
        # Hạn lưu theo NGÀY của từng camera (0 = không giới hạn). Sống trên
        # bảng `cameras` của engine chứ không ở bảng riêng: camera bị xoá là
        # hạn của nó biến mất theo, không để lại hàng mồ côi. Engine C++ liệt
        # kê cột tường minh trong mọi câu SELECT/INSERT nên cột này vô hình với
        # nó — chỉ bộ dọn của Python đọc.
        await conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS cameras "
            "ADD COLUMN IF NOT EXISTS retention_days INTEGER NOT NULL DEFAULT 0"
        )
        # Có ghi sự kiện của AI này xuống DB/đĩa hay không, theo TỪNG CAMERA và
        # TỪNG LOẠI AI. DEFAULT true: mọi cấu hình đã có từ trước phải giữ
        # nguyên hành vi cũ, tắt là việc người dùng chủ động chọn.
        await conn.exec_driver_sql(
            "ALTER TABLE IF EXISTS ai_configs "
            "ADD COLUMN IF NOT EXISTS save_events BOOLEAN NOT NULL DEFAULT true"
        )
    # Prime the plate whitelist cache before the ALPR consumer thread
    # starts so the very first detection can hit the in-memory map.
    async with AsyncSessionLocal() as db:
        await plate_white_list_service.load_all(db)
        await plate_white_list_settings_service.load_all(db)
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
    # Tự dọn dung lượng: giữ tối thiểu N GB trống, xoá dữ liệu cũ khi cần.
    threading.Thread(target=task_storage_cleanup.worker, daemon=True).start()
    threading.Thread(target=play_sound.play_sound, daemon=True).start()
    # Duong xem thu hai: MoQ tren QUIC/WebTransport, song song WebRTC. Chay
    # event loop RIENG de mot nguoi xem mang cham khong lam cham REST/WS.
    threading.Thread(target=moq_server.worker, daemon=True).start()
    yield
    process_ai_service.stop()
    task_system_metrics.stop()
    task_storage_cleanup.stop()
    moq_server.stop()
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
