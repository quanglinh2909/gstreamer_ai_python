from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.system_metrics_repository import SystemMetricsRepository
from app.utils import system_metrics_reader as reader


class SystemMetricsService:
    async def fetch_all(
        self,
        db: AsyncSession,
        limit: int,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
    ) -> dict:
        """Both the current snapshot and the recent history of every series."""
        current = await SystemMetricsRepository.fetch_latest(db)
        # Disk is live-only (no history table) — read it fresh for `current`.
        disk = reader.read_disk()
        disk["ts"] = reader.now_ts()
        current["disk"] = disk
        history = await SystemMetricsRepository.fetch_all(db, limit, from_ts, to_ts)
        return {"current": current, "history": history}


system_metrics_service = SystemMetricsService()
