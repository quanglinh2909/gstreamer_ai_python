# -*- coding: utf-8 -*-
"""Nền tự dọn dung lượng: mỗi CHECK_INTERVAL giây đo chỗ trống ổ đĩa; nếu dưới
ngưỡng thì xoá dữ liệu cũ (xem storage_cleanup_service). Chạy trên thread + event
loop riêng với async engine riêng (asyncpg buộc connection theo loop tạo ra nó,
giống task_system_metrics / task_parking_lot).
"""

import asyncio
import time
import traceback

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.services.storage_cleanup_service import storage_cleanup_service


class TaskStorageCleanup:
    CHECK_INTERVAL = 60  # giây giữa hai lần kiểm tra chỗ trống

    def __init__(self):
        self._running = True

    def worker(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(f"task_storage_cleanup worker crashed: {exc}")
            traceback.print_exc()

    def stop(self):
        self._running = False

    async def _run(self):
        engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
        session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        try:
            while self._running:
                start = time.time()
                try:
                    async with session_factory() as session:
                        stats = await storage_cleanup_service.run_once(session)
                    if stats.deleted_rows:
                        gb = stats.freed_bytes / (1024 ** 3)
                        parts = ", ".join(
                            f"{k}:{v / (1024 ** 3):.2f}GB"
                            for k, v in stats.per_category.items()
                        )
                        print(f"[storage] da xoa {stats.deleted_rows} hang, "
                              f"giai phong {gb:.2f}GB ({parts})")
                except Exception as e:
                    print(f"[storage] cycle error: {e}")
                    traceback.print_exc()
                # Ngủ phần còn lại của chu kỳ, thức sớm nếu được yêu cầu dừng.
                elapsed = time.time() - start
                for _ in range(int(max(0, self.CHECK_INTERVAL - elapsed))):
                    if not self._running:
                        break
                    await asyncio.sleep(1)
        finally:
            await engine.dispose()


task_storage_cleanup = TaskStorageCleanup()
