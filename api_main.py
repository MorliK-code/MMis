"""Run MMis API server."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys

import uvicorn

from config.settings import load_config, setup_logging


def _is_port_in_use(host: str, port: int) -> bool:
    """Проверяет, занят ли порт."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _kill_processes_on_port(port: int) -> int:
    """Завершает процессы, использующие указанный порт."""
    if os.name != "nt":
        return 0
    
    killed = 0
    try:
        # Находим PID процесса на порту
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0
        )
        
        for line in result.stdout.splitlines():
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = int(parts[-1])
                    if pid > 0 and pid != os.getpid():
                        try:
                            subprocess.run(
                                ['taskkill', '/F', '/PID', str(pid)],
                                capture_output=True,
                                text=True,
                                check=False,
                                timeout=3.0
                            )
                            killed += 1
                        except Exception:
                            pass
    except Exception:
        pass
    
    return killed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMis API server")
    parser.add_argument("--mmis-tag", default="mmis", help="MMis API process tag for cleanup")
    parser.add_argument(
        "--no-clean-tagged",
        action="store_true",
        help="Do not cleanup previously tagged MMis API processes before startup",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = load_config()
    setup_logging(cfg)
    
    # Проверяем, занят ли порт
    if _is_port_in_use(cfg.host, cfg.port):
        print(f"⚠️  Port {cfg.port} is in use, cleaning up...")
        killed = _kill_processes_on_port(cfg.port)
        if killed > 0:
            print(f"✅ Terminated {killed} process(es) on port {cfg.port}")
            import time
            time.sleep(1)  # Ждём освобождения порта
    
    # Start API server
    print(f"🚀 Starting API server on {cfg.host}:{cfg.port}...")
    uvicorn.run("api.app:app", host=cfg.host, port=cfg.port, reload=False)
