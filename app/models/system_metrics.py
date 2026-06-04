"""Time-series tables for on-device resource monitoring.

Six independent tables — one per metric family — each sampled on a fixed
interval by ``app.tasks.task_system_metrics`` and pruned to a 30-day window.
Every row carries an epoch-second ``ts`` (indexed) so range queries and the
retention sweep are cheap. Values that can't be read on a given cycle (e.g.
NPU/RGA debugfs without permission) are stored as NULL rather than dropped.
"""

from typing import Optional

from sqlalchemy import BigInteger, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class CpuUsageMetric(Base):
    """Overall CPU utilisation (%) plus the raw per-core breakdown."""

    __tablename__ = "metric_cpu_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    usage_percent: Mapped[float] = mapped_column(nullable=False)
    # JSON-encoded list of per-core percentages, e.g. "[12.5, 3.0, ...]".
    per_core: Mapped[Optional[str]] = mapped_column(nullable=True)

    __table_args__ = (Index("ix_metric_cpu_usage_ts", "ts"),)


class CpuTemperatureMetric(Base):
    """Per-thermal-zone temperatures (°C) reported by the RK3588 SoC."""

    __tablename__ = "metric_cpu_temperature"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # soc-thermal is the headline CPU temperature; the rest are kept for detail.
    soc_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    bigcore0_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    bigcore1_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    littlecore_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    center_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    gpu_c: Mapped[Optional[float]] = mapped_column(nullable=True)
    npu_c: Mapped[Optional[float]] = mapped_column(nullable=True)

    __table_args__ = (Index("ix_metric_cpu_temperature_ts", "ts"),)


class MemoryMetric(Base):
    """System RAM usage snapshot (bytes + percent)."""

    __tablename__ = "metric_memory"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    used_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    available_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    percent: Mapped[float] = mapped_column(nullable=False)

    __table_args__ = (Index("ix_metric_memory_ts", "ts"),)


class DiskMetric(Base):
    """Root filesystem (/) usage snapshot (bytes + percent)."""

    __tablename__ = "metric_disk"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    used_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    free_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    percent: Mapped[float] = mapped_column(nullable=False)

    __table_args__ = (Index("ix_metric_disk_ts", "ts"),)


class LoadAvgMetric(Base):
    """1/5/15-minute load averages, with the core count for normalisation."""

    __tablename__ = "metric_load_avg"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    load1: Mapped[float] = mapped_column(nullable=False)
    load5: Mapped[float] = mapped_column(nullable=False)
    load15: Mapped[float] = mapped_column(nullable=False)
    cpu_count: Mapped[Optional[int]] = mapped_column(nullable=True)

    __table_args__ = (Index("ix_metric_load_avg_ts", "ts"),)


class NpuMetric(Base):
    """Per-core NPU load (%) from /sys/kernel/debug/rknpu/load, plus the mean."""

    __tablename__ = "metric_npu"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    load_percent: Mapped[Optional[float]] = mapped_column(nullable=True)
    core0: Mapped[Optional[float]] = mapped_column(nullable=True)
    core1: Mapped[Optional[float]] = mapped_column(nullable=True)
    core2: Mapped[Optional[float]] = mapped_column(nullable=True)

    __table_args__ = (Index("ix_metric_npu_ts", "ts"),)


class RgaMetric(Base):
    """Per-scheduler RGA load (%) from /sys/kernel/debug/rkrga/load, plus the mean."""

    __tablename__ = "metric_rga"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    load_percent: Mapped[Optional[float]] = mapped_column(nullable=True)
    core0: Mapped[Optional[float]] = mapped_column(nullable=True)
    core1: Mapped[Optional[float]] = mapped_column(nullable=True)
    core2: Mapped[Optional[float]] = mapped_column(nullable=True)

    __table_args__ = (Index("ix_metric_rga_ts", "ts"),)
