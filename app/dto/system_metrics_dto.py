"""Response shapes for the single combined system-metrics query endpoint."""

import json
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class CpuUsageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    usage_percent: float
    per_core: Optional[List[float]] = None

    @field_validator("per_core", mode="before")
    @classmethod
    def _decode_per_core(cls, v):
        # Stored as a JSON string column; surface it as a real list.
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return None
        return v


class CpuTemperatureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    soc_c: Optional[float] = None
    bigcore0_c: Optional[float] = None
    bigcore1_c: Optional[float] = None
    littlecore_c: Optional[float] = None
    center_c: Optional[float] = None
    gpu_c: Optional[float] = None
    npu_c: Optional[float] = None


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    total_bytes: int
    used_bytes: int
    available_bytes: int
    percent: float


class DiskResponse(BaseModel):
    # Live-only (read at request/broadcast time) — no DB row, so no id.
    ts: int
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent: float


class LoadAvgResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    load1: float
    load5: float
    load15: float
    cpu_count: Optional[int] = None


class NpuResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    load_percent: Optional[float] = None
    core0: Optional[float] = None
    core1: Optional[float] = None
    core2: Optional[float] = None


class RgaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: int
    load_percent: Optional[float] = None
    core0: Optional[float] = None
    core1: Optional[float] = None
    core2: Optional[float] = None


class SystemMetricsCurrent(BaseModel):
    """The latest single sample of each series (None if not collected yet)."""

    cpu_usage: Optional[CpuUsageResponse] = None
    cpu_temperature: Optional[CpuTemperatureResponse] = None
    memory: Optional[MemoryResponse] = None
    disk: Optional[DiskResponse] = None
    load_avg: Optional[LoadAvgResponse] = None
    npu: Optional[NpuResponse] = None
    rga: Optional[RgaResponse] = None


class SystemMetricsHistory(BaseModel):
    """All six series, each newest-first."""

    cpu_usage: List[CpuUsageResponse]
    cpu_temperature: List[CpuTemperatureResponse]
    memory: List[MemoryResponse]
    load_avg: List[LoadAvgResponse]
    npu: List[NpuResponse]
    rga: List[RgaResponse]


class SystemMetricsResponse(BaseModel):
    """`current` = giá trị hiện tại của từng chỉ số; `history` = lịch sử."""

    current: SystemMetricsCurrent
    history: SystemMetricsHistory
