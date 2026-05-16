"""Run MMis API server."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import uvicorn

from config.settings import load_config, setup_logging


# Глобальный флаг для обработки сигналов
_shutdown_requested = False


def _handle_shutdown_signal(signum, frame):
    """Обработчик сигналов завершения."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nSignal {signum} received, shutting down...")


def _shutdown_memory_core_worker():
    """Принудительно останавливает memory_core worker."""
    try:
        from memory_core.adapter import _memory_core_adapter
        if _memory_core_adapter is not None:
            print("Closing memory_core adapter...")
            _memory_core_adapter.close()
            print("Memory_core adapter closed")
    except Exception as exc:
        print(f"Worker shutdown error: {exc}")


# Регистрируем обработчики сигналов для Windows и Unix
if os.name == "nt":
    # Windows - CTRL_BREAK_EVENT
    try:
        signal.signal(signal.SIGBREAK, _handle_shutdown_signal)
    except (AttributeError, ValueError):
        pass
# Unix - SIGTERM, SIGINT
signal.signal(signal.SIGTERM, _handle_shutdown_signal)
signal.signal(signal.SIGINT, _handle_shutdown_signal)


def _is_port_in_use(host: str, port: int) -> bool:
    """Проверяет, занят ли порт."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _cleanup_mmis_processes(tag: str = "mmis", *, quiet: bool = False) -> int:
    """
    Завершает все MMis API процессы с указанным тегом.

    Использует внешний скрипт stop_api.py для корректного завершения.
    """
    script_path = Path(__file__).parent / "stop_api.py"
    if not script_path.exists():
        if not quiet:
            print(f"stop_api.py not found at {script_path}")
        return 0

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--tag", tag, "--timeout", "15"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20.0,
        )
        # Выводим результат только если не quiet режим
        if not quiet:
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0 and result.stderr:
                print(f"Cleanup warning: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        if not quiet:
            print("Cleanup timed out")
        return 0
    except Exception as exc:
        if not quiet:
            print(f"Cleanup failed: {exc}")
        return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMis API server")
    parser.add_argument("--host", default=None, help="Host/interface to bind, e.g. 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=None, help="API port")
    parser.add_argument("--mmis-tag", default="mmis", help="MMis API process tag for cleanup")
    parser.add_argument(
        "--no-clean-tagged",
        action="store_true",
        help="Do not cleanup previously tagged MMis API processes before startup",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output (for embedded/automated usage)",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = load_config()
    overrides = {}
    if args.host:
        overrides["host"] = str(args.host)
    if args.port is not None:
        overrides["port"] = int(args.port)
    if overrides:
        cfg = replace(cfg, **overrides)
    setup_logging(cfg)

    quiet = bool(args.quiet)

    # Очищаем старые процессы перед запуском
    if not args.no_clean_tagged:
        if not quiet:
            print(f"Cleaning up existing MMis API processes (tag={args.mmis_tag})...")
        _cleanup_mmis_processes(tag=args.mmis_tag, quiet=quiet)
        time.sleep(0.5)

    # Проверяем, занят ли порт
    if _is_port_in_use(cfg.host, cfg.port):
        if not quiet:
            print(f"Port {cfg.port} is in use, forcing cleanup...")
        _cleanup_mmis_processes(tag=args.mmis_tag, quiet=quiet)
        time.sleep(1.5)  # Ждём освобождения порта

    # Start API server
    if not quiet:
        print(f"Starting API server on {cfg.host}:{cfg.port}...")
        print("Press Ctrl+C to stop the server and shutdown all workers...")
        print()

    # Запускаем uvicorn с обработкой сигналов
    # Uvicorn автоматически обрабатывает SIGINT/SIGTERM и вызывает lifespan shutdown
    try:
        uvicorn.run(
            "api.app:app",
            host=cfg.host,
            port=cfg.port,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        if not quiet:
            print("\nCtrl+C received, shutting down gracefully...")
    except Exception as exc:
        if not quiet:
            import traceback
            print(f"\nAPI error: {exc}")
            print("Traceback:")
            traceback.print_exc()
    finally:
        # Принудительно останавливаем worker если он ещё работает
        _shutdown_memory_core_worker()
        if not quiet:
            print("API server stopped")
