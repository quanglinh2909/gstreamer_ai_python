"""Background collector: sample the six resource metrics every SAMPLE_INTERVAL
seconds, broadcast them to WebSocket subscribers, persist one row per series,
and prune anything older than RETENTION_DAYS so each table stays a rolling
~1-month window.

Each tick reads every source exactly once; that single reading feeds both the
WebSocket push and the DB insert, so the live feed and the stored history stay
in lock-step at the same cadence.

Runs on its own thread/event loop with a dedicated async engine — asyncpg ties
connections to the loop that created them, so it cannot share the app's global
engine (same constraint as task_parking_lot).
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
from app.repositories.system_metrics_repository import SystemMetricsRepository
from app.utils import system_metrics_reader as reader
from app.ws.system_metrics_ws import system_metrics_broadcaster


class TaskSystemMetrics:
    SAMPLE_INTERVAL = 10          # giây giữa hai lần lấy mẫu (WS + lưu DB cùng nhịp)
    RETENTION_DAYS = 30           # giữ dữ liệu trong 1 tháng
    # Chạy dọn dữ liệu cũ mỗi giờ thay vì mỗi chu kỳ (360 chu kỳ * 10s = 1h).
    PURGE_EVERY_CYCLES = 360

    def __init__(self):
        self._session_factory = None
        self._running = True

    def worker(self):
        # Thread entrypoint: drive the async loop on a fresh event loop.
        try:
            asyncio.run(self._run())
        except Exception as exc:
            print(f"task_system_metrics worker crashed: {exc}")
            traceback.print_exc()

    def stop(self):
        self._running = False

    async def _run(self):
        engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)
        self._session_factory = async_sessionmaker(
            bind=engine, class_=AsyncSession, expire_on_commit=False,
        )
        # Prime cpu_percent so the first persisted sample reflects a real
        # window rather than the 0.0 the very first call always returns.
        reader.read_cpu_usage()
        cycle = 0
        try:
            while self._running:
                start = time.time()
                try:
                    await self._collect_once()
                    if cycle % self.PURGE_EVERY_CYCLES == 0:
                        await self._purge_old()
                except Exception as e:
                    print(f"[metrics] cycle error: {e}")
                    traceback.print_exc()
                cycle += 1
                # Keep a steady cadence regardless of how long the cycle took.
                elapsed = time.time() - start
                await asyncio.sleep(max(0.0, self.SAMPLE_INTERVAL - elapsed))
        finally:
            await engine.dispose()

    async def _collect_once(self):
        # Reads are blocking sysfs/debugfs/psutil calls — push off the loop.
        # One read per tick feeds both the WS broadcast and the DB insert.
        m = await asyncio.to_thread(reader.collect_all)

        # Push the live frame first so subscribers see it even if the DB
        # write is slow or fails.
        system_metrics_broadcaster.publish({"type": "system_metrics", **m})

        async with self._session_factory() as db:
            await SystemMetricsRepository.insert_sample(
                db, m["ts"], m["cpu_usage"], m["cpu_temperature"],
                m["memory"], m["disk"], m["load_avg"], m["npu"], m["rga"],
            )

    async def _purge_old(self):
        cutoff = reader.now_ts() - self.RETENTION_DAYS * 86400
        async with self._session_factory() as db:
            await SystemMetricsRepository.purge_older_than(db, cutoff)


task_system_metrics = TaskSystemMetrics()
