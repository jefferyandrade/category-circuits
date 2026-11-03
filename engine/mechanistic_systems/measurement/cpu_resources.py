"""CPU availability for host work under process and cgroup limits."""
from __future__ import annotations

import math
import os
from pathlib import Path


def _cgroup_cpu_limit(
    proc_self_cgroup: str = "/proc/self/cgroup",
    cgroup_root: str = "/sys/fs/cgroup",
) -> int | None:
    """Return the tightest CPU quota across the process's cgroup ancestors."""
    root = Path(cgroup_root)
    try:
        memberships = Path(proc_self_cgroup).read_text().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for membership in memberships:
        fields = membership.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        v2 = not controllers
        if not v2 and "cpu" not in controllers.split(","):
            continue
        bases = (root,) if v2 else (root / "cpu", root / "cpu,cpuacct", root)
        relative_path = Path(relative.lstrip("/"))
        if ".." in relative_path.parts:
            continue
        for base in bases:
            current = base / relative_path
            while True:
                try:
                    if v2:
                        quota, period = (current / "cpu.max").read_text().split()[:2]
                        quota_value = -1 if quota == "max" else int(quota)
                        period_value = int(period)
                    else:
                        quota_value = int((current / "cpu.cfs_quota_us").read_text())
                        period_value = int((current / "cpu.cfs_period_us").read_text())
                    if quota_value > 0 and period_value > 0:
                        limits.append(max(1, math.ceil(quota_value / period_value)))
                except (OSError, ValueError):
                    pass
                if current == base:
                    break
                current = current.parent
    return min(limits) if limits else None


def _available_cpu_count() -> int:
    """Count usable cores with the current affinity and inherited CPU quota."""
    try:
        available = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        available = max(1, os.cpu_count() or 1)
    quota = _cgroup_cpu_limit()
    return available if quota is None else min(available, quota)
