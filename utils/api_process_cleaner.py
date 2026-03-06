from __future__ import annotations

import os
import subprocess
from pathlib import Path


def clean_mmis_api_processes(
    *,
    tag: str = "mmis",
    root: str | Path | None = None,
    exclude_pid: int | None = None,
    timeout_sec: float = 12.0,
) -> int:
    """Best-effort cleanup of stale MMis API processes on Windows."""
    if os.name != "nt":
        return 0

    base = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    script = base / "scripts" / "clean_mmis_apis.ps1"
    if not script.exists():
        return 0

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Tag",
        str(tag or "mmis").strip() or "mmis",
        "-Root",
        str(base),
        "-ExcludePid",
        str(int(exclude_pid) if exclude_pid is not None else 0),
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(2.0, float(timeout_sec)),
        )
    except Exception:
        return 0

    if proc.returncode != 0:
        return 0

    killed = 0
    for line in str(proc.stdout or "").splitlines():
        row = str(line or "").strip()
        if row.startswith("KILLED_COUNT="):
            try:
                return max(0, int(row.split("=", 1)[1].strip()))
            except Exception:
                continue
        if row.startswith("KILLED:"):
            killed += 1
    return killed
