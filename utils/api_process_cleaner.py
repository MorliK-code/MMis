from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


def clean_mmis_api_processes(
    *,
    tag: str = "mmis",
    root: str | Path | None = None,
    exclude_pid: int | None = None,
    timeout_sec: float = 12.0,
) -> int:
    """Best-effort cleanup of stale MMis API processes on Windows."""
    killed = 0
    
    if os.name == "nt":  # Windows
        # Используем taskkill для поиска и завершения процессов
        try:
            # Ищем процессы python с api_main.py и --mmis-tag
            result = subprocess.run(
                ['tasklist', '/FO', 'CSV', '/NH'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0
            )
            
            for line in result.stdout.splitlines():
                parts = [p.strip('"').strip() for p in line.split(',')]
                if len(parts) >= 2 and 'python' in parts[0].lower():
                    pid = int(parts[1])
                    if pid <= 0 or pid == exclude_pid or pid == os.getpid():
                        continue
                    
                    # Проверяем командную строку
                    try:
                        cmd_result = subprocess.run(
                            ['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'CommandLine'],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=3.0
                        )
                        cmd = cmd_result.stdout
                        if not cmd:
                            continue
                        
                        cmd_norm = cmd.lower()
                        tag_norm = tag.lower()
                        
                        # Проверяем наличие тега или api_main.py
                        is_tagged = f'--mmis-tag' in cmd_norm and tag_norm in cmd_norm
                        is_api_main = 'api_main.py' in cmd_norm
                        is_main_api = 'main.py' in cmd_norm and '--mode' in cmd_norm and 'api' in cmd_norm
                        
                        if is_tagged or is_api_main or is_main_api:
                            # Завершаем процесс
                            kill_result = subprocess.run(
                                ['taskkill', '/F', '/PID', str(pid)],
                                capture_output=True,
                                text=True,
                                check=False,
                                timeout=3.0
                            )
                            if kill_result.returncode == 0:
                                killed += 1
                    except Exception:
                        continue
        except Exception:
            pass
        
        # Ждём немного чтобы процессы успели завершиться
        if killed > 0:
            time.sleep(0.5)
        
        return killed
    
    else:  # Linux/Mac
        try:
            subprocess.run(
                ["pkill", "-f", f"api_main.py.*--mmis-tag.*{tag}"],
                check=False,
                capture_output=True,
                timeout=max(2.0, float(timeout_sec)),
            )
            return 1
        except Exception:
            return 0
