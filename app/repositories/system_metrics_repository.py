"""Persistence for the six metric series: bulk insert, range query, prune."""

import json
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_metrics import (
    CpuTemperatureMetric,
    CpuUsageMetric,
    DiskMetric,
    LoadAvgMetric,
    MemoryMetric,
    NpuMetric,
    RgaMetric,
)

# series key -> ORM model. The query/cleanup paths iterate this so adding a
# series only means adding a model here (and a reader call in the collector).
METRIC_MODELS = {
    "cpu_usage": CpuUsageMetric,
    "cpu_temperature": CpuTemperatureMetric,
    "memory": MemoryMetric,
    "disk": DiskMetric,
    "load_avg": LoadAvgMetric,
    "npu": NpuMetric,
    "rga": RgaMetric,
}


class SystemMetricsRepository:
    @staticmethod
    async def insert_sample(
        db: AsyncSession,
        ts: int,
        cpu_usage: dict,
        cpu_temperature: dict,
        memory: dict,
        disk: dict,
        load_avg: dict,
        npu: dict,
        rga: dict,
    ) -> None:
        """Write one row into each of the seven tables for a single cycle."""
        db.add(CpuUsageMetric(
            ts=ts,
            usage_percent=cpu_usage["usage_percent"],
            per_core=(
                json.dumps(cpu_usage["per_core"])
                if cpu_usage.get("per_core") is not None else None
            ),
        ))
        db.add(CpuTemperatureMetric(ts=ts, **cpu_temperature))
        db.add(MemoryMetric(ts=ts, **memory))
        db.add(DiskMetric(ts=ts, **disk))
        db.add(LoadAvgMetric(ts=ts, **load_avg))
        db.add(NpuMetric(ts=ts, **npu))
        db.add(RgaMetric(ts=ts, **rga))
        await db.commit()

    @staticmethod
    async def purge_older_than(db: AsyncSession, cutoff_ts: int) -> None:
        """Delete rows in every series with ts < cutoff_ts (retention sweep)."""
        for model in METRIC_MODELS.values():
            await db.execute(delete(model).where(model.ts < cutoff_ts))
        await db.commit()

    @staticmethod
    async def fetch_latest(db: AsyncSession) -> dict:
        """Most-recent single row of every series (the 'current' snapshot).

        Independent of any range filter — always the absolute latest sample,
        at most one sampling interval old. None for a series with no rows yet.
        """
        out = {}
        for key, model in METRIC_MODELS.items():
            stmt = select(model).order_by(model.ts.desc(), model.id.desc()).limit(1)
            out[key] = (await db.execute(stmt)).scalars().first()
        return out

    @staticmethod
    async def fetch_all(
        db: AsyncSession,
        limit: int,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
    ) -> dict:
        """Return up to `limit` most-recent rows for every series, newest first.

        Optional from_ts/to_ts bound the window (inclusive). The result is a
        dict keyed by series name so a single response carries all six metrics.
        """
        out = {}
        for key, model in METRIC_MODELS.items():
            stmt = select(model)
            if from_ts is not None:
                stmt = stmt.where(model.ts >= from_ts)
            if to_ts is not None:
                stmt = stmt.where(model.ts <= to_ts)
            stmt = stmt.order_by(model.ts.desc(), model.id.desc()).limit(limit)
            rows = (await db.execute(stmt)).scalars().all()
            out[key] = rows
        return out
