"""System metrics for the /metrics endpoint — same JSON shape vre-video-worker exposes so the admin
dashboard's GENERIC Servers panel renders this worker with no dashboard changes. Pure stdlib (Linux
/proc + shutil); every probe is guarded so a non-Linux dev box degrades to nulls instead of crashing.
The dashboard card reads: cpu.{cores,percent,load1,loadPerCore}, mem.usedPct, disk.usedPct, uptimeSec.
"""
import os
import shutil
import socket
import time

_PROC_START = time.monotonic()  # process uptime baseline (system uptime comes from /proc/uptime)


def _cpu_percent(sample_ms=120):
    """Instantaneous CPU% across all cores: sample /proc/stat idle-vs-total twice, diff (matches the
    video worker's cpuPercent). None on any non-Linux / read error."""
    def snap():
        with open("/proc/stat") as f:
            vals = [int(x) for x in f.readline().split()[1:]]  # user nice system idle iowait irq ...
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)     # idle + iowait
        return idle, sum(vals)
    try:
        i1, t1 = snap()
        time.sleep(sample_ms / 1000.0)
        i2, t2 = snap()
        dt = t2 - t1
        return max(0.0, min(100.0, 100.0 * (1 - (i2 - i1) / dt))) if dt > 0 else 0.0
    except Exception:
        return None


def _mem():
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = int(v.strip().split()[0]) * 1024  # kB → bytes
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - avail
        return {"totalBytes": total, "usedBytes": used,
                "usedPct": round(100 * used / total) if total else None}
    except Exception:
        return None


def _disk():
    try:
        u = shutil.disk_usage("/")
        return {"totalBytes": u.total, "usedBytes": u.used,
                "usedPct": round(100 * u.used / u.total) if u.total else None}
    except Exception:
        return None


def _uptime_sec():
    try:
        with open("/proc/uptime") as f:
            return round(float(f.readline().split()[0]))
    except Exception:
        return None


def _rss_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        pass
    return None


def collect():
    """→ dict matching vre-video-worker's /metrics `collect()` (minus its DB queue, which main.py adds)."""
    cores = os.cpu_count() or 1
    try:
        load = os.getloadavg()
    except (OSError, AttributeError):
        load = (0.0, 0.0, 0.0)
    cpu_pct = _cpu_percent()
    return {
        "ok": True,
        "service": "vre-nsfw-worker",
        "host": socket.gethostname(),
        "at": int(time.time() * 1000),
        "uptimeSec": _uptime_sec(),
        "cpu": {
            "cores": cores,
            "percent": round(cpu_pct) if cpu_pct is not None else None,
            "load1": round(load[0], 2),
            "load5": round(load[1], 2),
            "load15": round(load[2], 2),
            "loadPerCore": round(load[0] / cores, 2) if cores else None,
        },
        "mem": _mem(),
        "disk": _disk(),
        "process": {"rssBytes": _rss_bytes(), "uptimeSec": round(time.monotonic() - _PROC_START)},
    }
