"""Pure readers for the six on-device resource metrics.

Each ``read_*`` returns a plain dict of values (or ``None`` fields) and never
raises — a metric source that is missing or unreadable yields NULLs so the
collector can still persist the row. CPU/memory/load/thermal come from psutil
and sysfs (world-readable); NPU/RGA come from debugfs and require the process
to have read access (root or a sudoers/udev grant).
"""

import os
import re
import subprocess
import time
from typing import Optional

import psutil

# Maps thermal-zone `type` -> column name on CpuTemperatureMetric.
_THERMAL_TYPE_TO_FIELD = {
    "soc-thermal": "soc_c",
    "bigcore0-thermal": "bigcore0_c",
    "bigcore1-thermal": "bigcore1_c",
    "littlecore-thermal": "littlecore_c",
    "center-thermal": "center_c",
    "gpu-thermal": "gpu_c",
    "npu-thermal": "npu_c",
}

_NPU_LOAD_PATH = "/sys/kernel/debug/rknpu/load"
_RGA_LOAD_PATH = "/sys/kernel/debug/rkrga/load"


def now_ts() -> int:
    return int(time.time())


def read_cpu_usage() -> dict:
    """Overall + per-core CPU utilisation since the previous call.

    Uses ``interval=None`` so the percentage is computed against the last
    invocation — the collector calls this once per cycle, so each value
    reflects the whole sampling window. The first call after process start
    returns 0.0 (no prior reference point), which is expected.
    """
    try:
        per_core = psutil.cpu_percent(interval=None, percpu=True)
        overall = sum(per_core) / len(per_core) if per_core else 0.0
        return {"usage_percent": round(overall, 2), "per_core": per_core}
    except Exception as e:
        print(f"[metrics] cpu usage read error: {e}")
        return {"usage_percent": 0.0, "per_core": None}


def read_cpu_temperature() -> dict:
    """Read every /sys/class/thermal zone we recognise, in °C."""
    fields = {f: None for f in _THERMAL_TYPE_TO_FIELD.values()}
    base = "/sys/class/thermal"
    try:
        for zone in os.listdir(base):
            if not zone.startswith("thermal_zone"):
                continue
            zdir = os.path.join(base, zone)
            try:
                with open(os.path.join(zdir, "type")) as f:
                    ztype = f.read().strip()
                field = _THERMAL_TYPE_TO_FIELD.get(ztype)
                if not field:
                    continue
                with open(os.path.join(zdir, "temp")) as f:
                    milli = int(f.read().strip())
                fields[field] = round(milli / 1000.0, 1)
            except Exception:
                continue
    except Exception as e:
        print(f"[metrics] cpu temperature read error: {e}")
    return fields


def read_memory() -> dict:
    try:
        vm = psutil.virtual_memory()
        return {
            "total_bytes": int(vm.total),
            "used_bytes": int(vm.used),
            "available_bytes": int(vm.available),
            "percent": float(vm.percent),
        }
    except Exception as e:
        print(f"[metrics] memory read error: {e}")
        return {"total_bytes": 0, "used_bytes": 0, "available_bytes": 0, "percent": 0.0}


def read_disk(path: str = "/") -> dict:
    """Disk usage of the filesystem holding `path` (default root /)."""
    try:
        du = psutil.disk_usage(path)
        return {
            "total_bytes": int(du.total),
            "used_bytes": int(du.used),
            "free_bytes": int(du.free),
            "percent": float(du.percent),
        }
    except Exception as e:
        print(f"[metrics] disk read error: {e}")
        return {"total_bytes": 0, "used_bytes": 0, "free_bytes": 0, "percent": 0.0}


def read_uptime() -> dict:
    """System uptime since boot (seconds) plus the boot epoch.

    Live-only — not persisted; surfaced in the API `current` snapshot and the
    WebSocket feed so a UI can show "đã chạy bao lâu".
    """
    try:
        boot = psutil.boot_time()
        return {"uptime_seconds": int(time.time() - boot), "boot_time": int(boot)}
    except Exception as e:
        print(f"[metrics] uptime read error: {e}")
        return {"uptime_seconds": 0, "boot_time": 0}


def read_load_avg() -> dict:
    try:
        l1, l5, l15 = os.getloadavg()
        return {
            "load1": round(l1, 2),
            "load5": round(l5, 2),
            "load15": round(l15, 2),
            "cpu_count": os.cpu_count(),
        }
    except Exception as e:
        print(f"[metrics] load avg read error: {e}")
        return {"load1": 0.0, "load5": 0.0, "load15": 0.0, "cpu_count": os.cpu_count()}


def _parse_core_percents(text: str) -> list:
    """Pull every ``NN%`` out of a debugfs load dump, in order."""
    return [float(m) for m in re.findall(r"(\d+)\s*%", text)]


def _read_debugfs(path: str) -> Optional[str]:
    """Return the contents of a debugfs load file, or None if unreadable.

    Tries a plain read first (works if the file/dir is world-readable, e.g.
    relaxed at boot). Falls back to ``sudo -n cat`` so a passwordless sudoers
    grant for these two paths also works without running the app as root.
    """
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None
    except PermissionError:
        pass
    except Exception as e:
        print(f"[metrics] {path} read error: {e}")
        return None
    # PermissionError fallthrough -> try sudo (NOPASSWD), stay non-blocking.
    try:
        res = subprocess.run(
            ["sudo", "-n", "cat", path],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None


def _read_load_file(path: str) -> dict:
    out = {"load_percent": None, "core0": None, "core1": None, "core2": None}
    text = _read_debugfs(path)
    if not text:
        return out
    cores = _parse_core_percents(text)
    if cores:
        for i, v in enumerate(cores[:3]):
            out[f"core{i}"] = v
        out["load_percent"] = round(sum(cores) / len(cores), 2)
    return out


def read_npu() -> dict:
    """Per-core NPU load from rknpu debugfs. NULLs if unreadable.

    Expected content, e.g.:
        NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,
    """
    return _read_load_file(_NPU_LOAD_PATH)


def read_rga() -> dict:
    """Per-scheduler RGA load from rkrga debugfs. NULLs if unreadable.

    Layout varies by kernel; we average every percentage found rather than
    rely on a fixed line format.
    """
    return _read_load_file(_RGA_LOAD_PATH)


def collect_all() -> dict:
    """Read every metric once and return a single dict.

    Used by the collector so a tick reads each source exactly once (avoids
    two psutil callers fighting over cpu_percent's shared reference) — the
    same readings feed both the WebSocket broadcast and the DB insert.
    """
    return {
        "ts": now_ts(),
        "cpu_usage": read_cpu_usage(),
        "cpu_temperature": read_cpu_temperature(),
        "memory": read_memory(),
        "disk": read_disk(),
        "uptime": read_uptime(),
        "load_avg": read_load_avg(),
        "npu": read_npu(),
        "rga": read_rga(),
    }
