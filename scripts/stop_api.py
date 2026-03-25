#!/usr/bin/env python3
"""
Скрипт для корректного завершения MMis API сервера и всех worker процессов.

Использование:
    python stop_api.py [--tag mmis] [--timeout 10]

Аргументы:
    --tag       Тег MMis API процесса (по умолчанию: mmis)
    --timeout   Таймаут ожидания завершения в секундах (по умолчанию: 10)
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def get_process_command_line(pid: int) -> str:
    """Получает командную строку процесса по PID."""
    if os.name == "nt":  # Windows
        try:
            result = subprocess.run(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/value"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
            # Формат: CommandLine=...\python.exe  api_main.py --mmis-tag mmis
            for line in result.stdout.splitlines():
                if line.startswith("CommandLine="):
                    return line.split("=", 1)[1] if "=" in line else ""
            return ""
        except Exception:
            return ""
    else:  # Linux/Mac
        try:
            with open(f"/proc/{pid}/cmdline", "r", encoding="utf-8") as f:
                return f.read().replace("\x00", " ")
        except Exception:
            return ""
    return ""


def find_mmis_processes(tag: str = "mmis") -> List[Tuple[int, str]]:
    """
    Находит все процессы MMis API.

    Returns:
        Список кортежей (pid, command_line).
    """
    processes = []
    current_pid = os.getpid()

    if os.name == "nt":  # Windows
        try:
            # Получаем список всех Python процессов
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )

            for line in result.stdout.splitlines():
                parts = [p.strip('"').strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        continue

                    if pid <= 0 or pid == current_pid:
                        continue

                    cmd = get_process_command_line(pid)
                    if not cmd:
                        continue

                    cmd_lower = cmd.lower()
                    tag_lower = tag.lower()

                    # Проверяем, является ли процесс MMis API
                    is_mmis_api = (
                        ("api_main.py" in cmd_lower) or
                        ("main.py" in cmd_lower and "--mode" in cmd_lower and "api" in cmd_lower) or
                        ("--mmis-tag" in cmd_lower and tag_lower in cmd_lower)
                    )

                    if is_mmis_api:
                        processes.append((pid, cmd.strip()))

        except Exception as exc:
            print(f"⚠️  Error finding processes: {exc}", file=sys.stderr)

    else:  # Linux/Mac
        try:
            result = subprocess.run(
                ["pgrep", "-f", "python.*api_main.py"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )

            for line in result.stdout.splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue

                if pid <= 0 or pid == current_pid:
                    continue

                cmd = get_process_command_line(pid)
                if cmd:
                    processes.append((pid, cmd.strip()))

        except Exception as exc:
            print(f"⚠️  Error finding processes: {exc}", file=sys.stderr)

    return processes


def request_llm_unload(api_port: int = 8000) -> bool:
    """
    Отправляет HTTP запрос для принудительной выгрузки LLM из VRAM.
    
    Args:
        api_port: Порт API сервера.
    
    Returns:
        True если успешно.
    """
    import urllib.request
    import urllib.error
    
    url = f"http://localhost:{api_port}/unload-llm"
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            if response.status == 200:
                print(f"✅ LLM unload request sent (port {api_port})")
                return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"ℹ️  LLM unload endpoint not available (port {api_port})")
        else:
            print(f"⚠️  LLM unload failed (port {api_port}): {e.code}")
    except Exception as exc:
        print(f"⚠️  LLM unload error (port {api_port}): {exc}")
    return False


def terminate_process(pid: int, timeout: float = 15.0) -> bool:
    """
    Корректно завершает процесс с долгим ожиданием cleanup.
    
    Отправляет мягкий сигнал и ждёт завершения worker.
    """
    if os.name == "nt":  # Windows
        # Пробуем мягкое завершение
        try:
            print(f"Sending graceful terminate to PID {pid}...")
            result = subprocess.run(
                ["taskkill", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
            
            if result.returncode == 0:
                # Ждём завершения процесса и cleanup worker
                print(f"Waiting up to {timeout}s for process and worker cleanup...")
                for i in range(int(timeout * 2)):
                    if not is_process_running(pid):
                        print(f"Process {pid} terminated gracefully after {i/2:.1f}s")
                        time.sleep(2.0)  # Дополнительная пауза для освобождения VRAM
                        return True
                    time.sleep(0.5)
                
                # Процесс не завершился сам
                print(f"Process {pid} did not terminate gracefully, forcing...")
            else:
                print(f"Graceful terminate failed: {result.stderr}")
        except Exception as exc:
            print(f"Error during graceful terminate: {exc}")
        
        # Принудительное завершение (последняя попытка)
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
                timeout=3.0,
            )
            if result.returncode == 0:
                print(f"Process {pid} terminated forcefully")
                # Всё равно ждём для освобождения ресурсов
                print(f"Waiting {timeout}s for resource cleanup...")
                time.sleep(timeout)
                return True
            else:
                print(f"Failed to force terminate: {result.stderr}")
                return False
        except Exception as exc:
            print(f"Error during force terminate: {exc}")
            return False
    
    else:  # Linux/Mac
        try:
            # SIGTERM
            os.kill(pid, signal.SIGTERM)
            
            # Ждём завершения
            for _ in range(int(timeout * 2)):
                if not is_process_running(pid):
                    return True
                time.sleep(0.5)
            
            # SIGKILL
            print(f"Process {pid} did not terminate, sending SIGKILL...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(timeout)
            return True
        except Exception:
            return False
    
    return False


def kill_ollama_processes() -> int:
    """
    Принудительно завершает все процессы Ollama для освобождения VRAM.
    
    Returns:
        Количество завершённых процессов.
    """
    killed = 0
    
    if os.name == "nt":  # Windows
        try:
            # Находим процессы ollama
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq ollama.exe"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            
            for line in result.stdout.splitlines():
                parts = [p.strip('"').strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        continue
                    
                    if pid <= 0:
                        continue
                    
                    print(f"Killing Ollama process PID {pid}...", end=" ")
                    try:
                        kill_result = subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=3.0,
                        )
                        if kill_result.returncode == 0:
                            print("OK")
                            killed += 1
                        else:
                            print(f"FAILED: {kill_result.stderr.strip()}")
                    except Exception as exc:
                        print(f"FAILED: {exc}")
        except Exception as exc:
            print(f"Error finding Ollama processes: {exc}")
    else:  # Linux/Mac
        try:
            subprocess.run(["pkill", "-9", "ollama"], check=False, timeout=5.0)
            killed = 1
            print("Ollama processes killed")
        except Exception as exc:
            print(f"Error killing Ollama: {exc}")
    
    if killed > 0:
        print(f"Waiting 5s for VRAM to be freed...")
        time.sleep(5.0)
    
    return killed


def is_process_running(pid: int) -> bool:
    """Проверяет, запущен ли процесс с указанным PID."""
    if os.name == "nt":  # Windows
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2.0,
            )
            return str(pid) in result.stdout
        except Exception:
            return False
    else:  # Linux/Mac
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Корректно завершает MMis API сервер и worker процессы"
    )
    parser.add_argument(
        "--tag",
        default="mmis",
        help="Тег MMis API процесса (по умолчанию: mmis)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Таймаут ожидания завершения в секундах (по умолчанию: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать процессы, не завершать"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительное завершение без graceful shutdown"
    )

    args = parser.parse_args()

    print(f"Searching for MMis API processes with tag '{args.tag}'...")
    processes = find_mmis_processes(tag=args.tag)

    if not processes:
        print("No MMis API processes found")
        # Всё равно пробуем выгрузить LLM и убить Ollama
        print("\nRequesting LLM unload via HTTP...")
        request_llm_unload(api_port=8000)
        time.sleep(2.0)
        
        print("\nKilling Ollama processes to free VRAM...")
        ollama_killed = kill_ollama_processes()
        if ollama_killed > 0:
            print(f"Killed {ollama_killed} Ollama process(es)")
        else:
            print("No Ollama processes found")
        return 0

    print(f"Found {len(processes)} process(es):")
    for pid, cmd in processes:
        # Обрезаем командную строку для отображения
        cmd_display = cmd[:80] + "..." if len(cmd) > 80 else cmd
        print(f"  PID {pid}: {cmd_display}")

    if args.dry_run:
        print("\nDry-run mode. Processes not terminated.")
        return 0

    # Сначала пробуем выгрузить LLM через HTTP запрос
    print("\nRequesting LLM unload via HTTP...")
    request_llm_unload(api_port=8000)
    time.sleep(2.0)  # Ждём выгрузки

    if args.force:
        # Принудительное завершение без graceful
        print(f"\nForce killing processes...")
        for pid, cmd in processes:
            print(f"  Force killing PID {pid}...", end=" ")
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3.0,
                )
                print("OK")
            except Exception as exc:
                print(f"FAILED: {exc}")
        time.sleep(args.timeout)
    else:
        print(f"\nTerminating processes (timeout={args.timeout}s)...")

        terminated = 0
        failed = 0

        for pid, cmd in processes:
            print(f"  Terminating PID {pid}...", end=" ")
            if terminate_process(pid, timeout=args.timeout):
                print("OK")
                terminated += 1
            else:
                print("FAILED")
                failed += 1

        print()
        print(f"Results:")
        print(f"  Terminated: {terminated}")
        if failed > 0:
            print(f"  Failed: {failed}")

        # Дополнительная пауза для освобождения порта
        if terminated > 0:
            print("\nWaiting for port to be freed...")
            time.sleep(1.0)

    # Принудительно завершаем процессы Ollama для освобождения VRAM
    print("\nKilling Ollama processes to free VRAM...")
    ollama_killed = kill_ollama_processes()
    if ollama_killed > 0:
        print(f"Killed {ollama_killed} Ollama process(es)")
    else:
        print("No Ollama processes found")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
