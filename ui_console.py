from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ui.api_client import ApiClient, ApiClientError
from config.settings import get_config_payload, load_config, update_config_values
from utils.api_process_cleaner import clean_mmis_api_processes


CONSOLE_BUILD_ID = "2026-02-28-r2"
MMIS_API_TAG = "mmis"
config = load_config()
_UI_RUNTIME_KEYS = {"mode_lock", "active_mode", "output_parameters", "output_summary"}

@dataclass
class ConsoleState:
    api: ApiClient
    store_turn: bool = True
    show_thinking: bool = True
    thinking_first: bool = True
    think_enabled: bool | None = None
    web_mode: str | None = None
    json_mode_enabled: bool | None = None
    online: bool = False
    auto_start_api: bool = True
    auto_start_ollama: bool = True
    api_process: subprocess.Popen | None = None
    ollama_process: subprocess.Popen | None = None
    prefs: dict = field(default_factory=dict)
    debug_memory: bool = False  # Показывать memory retrieval в real-time


def _ui_state_file_path() -> Path:
    base = Path(getattr(config, "memory_dir", Path("data/memory"))).expanduser()
    return base / "ui_console_state.json"


def _load_ui_state() -> dict:
    path = _ui_state_file_path()
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8-sig") or "{}")
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict = {}
    if isinstance(payload.get("show_thinking"), bool):
        out["show_thinking"] = bool(payload.get("show_thinking"))
    if isinstance(payload.get("thinking_first"), bool):
        out["thinking_first"] = bool(payload.get("thinking_first"))
    if isinstance(payload.get("store_turn"), bool):
        out["store_turn"] = bool(payload.get("store_turn"))
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        out["runtime"] = dict(runtime)
    return out


def _save_ui_state(state: ConsoleState) -> None:
    path = _ui_state_file_path()
    runtime = dict(state.prefs.get("runtime") or {}) if isinstance(state.prefs, dict) else {}
    payload = {
        "show_thinking": bool(state.show_thinking),
        "thinking_first": bool(state.thinking_first),
        "store_turn": bool(state.store_turn),
        "runtime": runtime,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _sync_ui_state_to_settings_file(state)
    except Exception:
        return


def _runtime_prefs(state: ConsoleState) -> dict:
    prefs = state.prefs if isinstance(state.prefs, dict) else {}
    runtime = prefs.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
        prefs["runtime"] = runtime
    state.prefs = prefs
    return runtime


def _sanitize_runtime_for_config(runtime: dict) -> dict:
    src = dict(runtime or {})
    out: dict = {}
    if isinstance(src.get("mode_lock"), bool):
        out["mode_lock"] = bool(src.get("mode_lock"))
    active_mode = str(src.get("active_mode") or "").strip()
    if active_mode:
        out["active_mode"] = active_mode
    if isinstance(src.get("output_parameters"), bool):
        out["output_parameters"] = bool(src.get("output_parameters"))
    if isinstance(src.get("output_summary"), bool):
        out["output_summary"] = bool(src.get("output_summary"))
    return out


def _set_runtime_pref(state: ConsoleState, key: str, value) -> None:
    runtime = _runtime_prefs(state)
    runtime[str(key)] = value
    _save_ui_state(state)
    _sync_runtime_pref_to_settings_file(str(key), value)


def _cfg_get_dotted(payload: dict, dotted: str, default=None):
    cur = dict(payload or {})
    for part in [x for x in str(dotted or "").split(".") if x]:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur.get(part)
    return cur


def _sync_ui_state_to_settings_file(state: ConsoleState) -> None:
    runtime = _sanitize_runtime_for_config(_runtime_prefs(state))
    updates = {
        "ui.console.store_turn": bool(state.store_turn),
        "ui.console.show_thinking": bool(state.show_thinking),
        "ui.console.thinking_first": bool(state.thinking_first),
        "ui.console.runtime": runtime,
    }
    try:
        update_config_values(updates)
    except Exception:
        return


def _sync_runtime_pref_to_settings_file(key: str, value) -> None:
    field = str(key or "").strip().lower()
    if not field:
        return
    updates: dict[str, object] = {}
    if field in _UI_RUNTIME_KEYS:
        updates[f"ui.console.runtime.{field}"] = value

    if field == "model":
        text = str(value or "").strip()
        if text:
            updates["llm.model_name"] = text
    elif field == "think_enabled":
        updates["llm.thinking_enabled"] = bool(value)
    elif field == "web_mode":
        text = str(value or "").strip().lower()
        if text in {"on", "off", "auto"}:
            updates["internet.web_mode"] = text
    elif field == "json_mode_enabled":
        flag = bool(value)
        updates["llm.json_mode_enabled"] = flag
    if not updates:
        return
    try:
        update_config_values(updates)
    except Exception:
        return


def _load_console_runtime_from_settings_file() -> dict:
    try:
        payload = get_config_payload(force_reload=True)
    except Exception:
        return {}
    runtime = _cfg_get_dotted(payload, "ui.console.runtime", {})
    return _sanitize_runtime_for_config(dict(runtime) if isinstance(runtime, dict) else {})


def _send_backend_command_silent(state: ConsoleState, command_text: str) -> bool:
    try:
        state.api.stream_chat(
            text=str(command_text or "").strip(),
            store_turn=False,
            think=bool(state.think_enabled),
            json_mode=False,
            on_chunk=None,
            on_thinking_chunk=None,
        )
        return True
    except ApiClientError:
        return False


def _apply_persisted_runtime_settings(state: ConsoleState) -> None:
    runtime = _runtime_prefs(state)

    mode_lock = runtime.get("mode_lock")
    if isinstance(mode_lock, bool):
        _send_backend_command_silent(state, f"/mode_lock {'on' if mode_lock else 'off'}")

    active_mode = str(runtime.get("active_mode") or "").strip()
    if active_mode:
        _send_backend_command_silent(state, f"/mode {active_mode}")

    output_parameters = runtime.get("output_parameters")
    if isinstance(output_parameters, bool):
        _send_backend_command_silent(state, f"/output parameters {'on' if output_parameters else 'off'}")

    output_summary = runtime.get("output_summary")
    if isinstance(output_summary, bool):
        _send_backend_command_silent(state, f"/output summary {'on' if output_summary else 'off'}")


def _configure_stdout() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def _build_parser() -> argparse.ArgumentParser:
    host = config.host
    port = config.port

    parser = argparse.ArgumentParser(description="MMis console chat (API client)")
    parser.add_argument("--api-url", default="", help=f"MMis API url (default: MMIS_API_URL or http://{host}:{port})")
    parser.add_argument("--model", default="", help="Set model on startup")
    parser.add_argument("--once", default="", help="Single message and exit")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout seconds")
    parser.add_argument("--stream-timeout", type=float, default=None, help="Stream timeout seconds")
    parser.add_argument("--no-store", action="store_true", help="Do not store turn in backend memory")
    think_view_group = parser.add_mutually_exclusive_group()
    think_view_group.add_argument("--show-thinking", action="store_true", help="Print thinking block")
    think_view_group.add_argument("--hide-thinking", action="store_true", help="Do not print thinking block")
    parser.add_argument("--no-auto-api", action="store_true", help="Do not auto-start API when offline")
    parser.add_argument("--no-auto-ollama", action="store_true", help="Do not auto-start Ollama when models backend is offline")

    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--think", action="store_true", help="Enable think mode on startup")
    think_group.add_argument("--nothink", action="store_true", help="Disable think mode on startup")
    think_order_group = parser.add_mutually_exclusive_group()
    think_order_group.add_argument("--thinking-first", action="store_true", help="Prefer streaming thinking before answer")
    think_order_group.add_argument("--thinking-last", action="store_true", help="Do not delay answer waiting for thinking")
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument("--json", action="store_true", help="Enable JSON mode on startup")
    json_group.add_argument("--nojson", action="store_true", help="Disable JSON mode on startup")
    return parser


def _print_header(state: ConsoleState) -> None:
    print("MMis Console UI")
    print(f"Build: {CONSOLE_BUILD_ID} (stream-output-render=on)")
    print(f"API: {state.api.base_url}")
    if _ensure_connected(state):
        print("Connected.")
        _print_health_short(state, startup=True)
        return

    print("API недоступен.")
    if state.auto_start_api:
        print("Пробую авто-запуск API...")
        if _start_api_process(state) and _wait_for_api(state, timeout_s=18.0):
            print("API поднят автоматически.")
            _print_health_short(state, startup=True)
            _ensure_model_backend(state)
            return
    print("Запусти вручную: python api_main.py --mmis-tag mmis")


def _print_help() -> None:
    print("Доступные команды:")
    print("")
    print("Локальные (UI):")
    print("/help                        показать эту справку")
    print("/connect [url]               переподключиться к API (можно указать URL)")
    print("/restartapi                  перезапустить локальный API-процесс")
    print("/restartall                  полный рестарт UI+API в этом же окне")
    print("/cleanapi                    убить все MMis API-процессы (tag=mmis)")
    print("/models                      показать доступные модели")
    print("/model                       показать текущую модель")
    print("/model <name>                переключить модель")
    print("/show-thinking               показывать поток thinking")
    print("/hide-thinking               скрывать поток thinking")
    print("/thinking-first              показывать thinking до ответа")
    print("/thinking-last               показывать ответ без ожидания thinking")
    print("/debug-memory on|off         показывать что находит память (real-time)")
    print("/store on|off                включить/выключить сохранение turns")
    print("/health                      проверить состояние API")
    print("/exit или /quit              выход")
    print("")
    print("Команды, отправляемые в backend:")
    print("/think                       включить thinking у модели")
    print("/nothink                     выключить thinking у модели")
    print("/web [query]                 enable web mode and optionally run query")
    print("/no-web                      выключить web-поиск")
    print("/output status               показать формат вывода")
    print("/output parameters on|off    включить/выключить блок [PARAMETERS]")
    print("/output summary on|off       включить/выключить блок [SUMMARY]")
    print("/mode <name>                 переключить backend mode (автоматически mode_lock=on)")
    print("/mode_lock on|off            включить/выключить lock mode")
    print("/brain_debug                 показать debug brain state")
    print("/persona_debug               показать debug persona state")
    print("/json                        включить JSON-режим ответа")
    print("/nojson                      выключить JSON-режим ответа")
    print("/character ...               команды управления персонажем")
    print("/character delete <id>       удалить персонажа (кроме default)")
    print("/trait ...                   команды управления traits")
    print("/studio ...                  единый модуль работы с персонажем (create/update/modes/mixed/build_pack)")
    print("/studio apply|cancel         локальные команды студии (не отправляются в основной чат-промпт)")
    print("/apply | /cancel             короткие локальные команды студии при active session")


def _print_health_short(state: ConsoleState, *, startup: bool = False) -> None:
    try:
        payload = state.api.health()
        state.online = True
        state.think_enabled = bool(payload.get("thinking_enabled", True))
        state.json_mode_enabled = bool(payload.get("json_mode_enabled", False))
        print(f"Model: {payload.get('model')}")
        if not startup:
            print(f"Thinking: {'on' if state.think_enabled else 'off'}")
            print(f"JSON mode: {'on' if state.json_mode_enabled else 'off'}")
            print(f"Thinking-first: {'on' if state.thinking_first else 'off'}")
    except ApiClientError as exc:
        state.online = False
        print(f"API error: {exc}")


def _handle_command(state: ConsoleState, line: str) -> bool:
    raw = str(line or "").strip()
    if not raw:
        return True
    cmd, *rest = raw.split(maxsplit=1)
    arg = rest[0].strip() if rest else ""
    key = cmd.lower()

    if key in {"/exit", "/quit"}:
        return False
    if key == "/help":
        _print_help()
        return True

    if key == "/connect":
        if arg:
            state.api.base_url = str(arg).strip().rstrip("/")
        if _ensure_connected(state):
            print("Connected.")
            _print_health_short(state)
            _ensure_model_backend(state)
            return True
        if state.auto_start_api and _start_api_process(state) and _wait_for_api(state, timeout_s=18.0):
            print("API started and connected.")
            _print_health_short(state)
            _ensure_model_backend(state)
            return True
        print("API is still offline.")
        return True

    if key in {"/restartapi", "/api-restart", "/restart-api"}:
        if _restart_api_process(state):
            print("API restarted.")
            _print_health_short(state)
            _ensure_model_backend(state)
        else:
            print("API restart failed.")
        return True

    if key in {"/restartall", "/restart-all", "/restart", "/restart-app", "/reboot"}:
        if _restart_full_app(state):
            print("Full app restarted (UI+API).")
            _print_health_short(state)
            _ensure_model_backend(state)
            return True
        print("Full app restart failed.")
        return True

    if key in {"/cleanapi", "/api-clean"}:
        stopped_model = _stop_runtime_ollama_model(state)
        stopped_memory_model = _stop_memory_ollama_model()
        stopped_owned_ollama = _stop_owned_ollama_process(state)
        killed = clean_mmis_api_processes(tag=MMIS_API_TAG, root=Path(__file__).resolve().parent)
        state.online = False
        state.api_process = None
        if stopped_model:
            print("Ollama runtime model unloaded.")
        if stopped_memory_model:
            print("Ollama memory model unloaded.")
        if stopped_owned_ollama:
            print("Owned Ollama process stopped.")
        print(f"MMis API cleanup: killed {killed} process(es).")
        return True

    if key == "/models":
        if not _ensure_connected_or_start(state):
            return True
        try:
            payload = state.api.list_models()
            models = [str(x) for x in list(payload.get("models") or []) if str(x).strip()]
            current = str(payload.get("runtime_model") or state.api.get_runtime_model() or "")
            if not models:
                print("No models returned by API.")
                return True
            print("Models:")
            for name in models:
                marker = "*" if current and name == current else " "
                print(f" {marker} {name}")
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True

    if key == "/model":
        if not _ensure_connected_or_start(state):
            return True
        if not arg:
            print(f"Current model: {state.api.get_runtime_model() or 'вЂ”'}")
            return True
        try:
            state.api.set_model(arg)
            print(f"Model switched to: {state.api.get_runtime_model() or arg}")
            _set_runtime_pref(state, "model", str(arg).strip())
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True

    if key in {"/think", "/nothink"}:
        if not _ensure_connected_or_start(state):
            return True

        target = (key == "/think")
        try:
            actual = bool(state.api.set_thinking_enabled(target))
            state.think_enabled = actual
            _set_runtime_pref(state, "think_enabled", bool(actual))
            print(f"Thinking: {'on' if actual else 'off'}")
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True
    
    if key in {"/show-thinking", "/hide-thinking"}:
        target = (key == "/show-thinking")
        state.show_thinking = target
        _save_ui_state(state)
        print(f"Thinking display: {'on' if target else 'off'}")
        return True

    if key in {"/thinking-first", "/thinking-last"}:
        state.thinking_first = (key == "/thinking-first")
        _save_ui_state(state)
        print(f"Thinking-first: {'on' if state.thinking_first else 'off'}")
        return True

    if key in {"/debug-memory", "/debug-memory-on", "/debug-memory-off"}:
        if key == "/debug-memory":
            state.debug_memory = not state.debug_memory
        elif key == "/debug-memory-on":
            state.debug_memory = True
        else:
            state.debug_memory = False
        _save_ui_state(state)
        print(f"Debug memory: {'on' if state.debug_memory else 'off'}")
        return True

    if key in {"/web", "/no-web"}:
        if not _ensure_connected_or_start(state):
            return True

        target = "on" if key == "/web" else "off"
        try:
            actual = str(state.api.set_web_mode(target))
            state.web_mode = actual
            _set_runtime_pref(state, "web_mode", str(actual).strip().lower())
            print(f"Web mode: {actual}")
            if key == "/web" and arg:
                _send_chat(state, arg)
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True
    
    if key in {"/json", "/nojson"}:
        if not _ensure_connected_or_start(state):
            return True
        target = key == "/json"
        try:
            actual = bool(state.api.set_json_mode_enabled(target))
            state.json_mode_enabled = actual
            _set_runtime_pref(state, "json_mode_enabled", bool(actual))
            print(f"JSON mode: {'on' if actual else 'off'}")
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True

    if key == "/health":
        if not _ensure_connected_or_start(state):
            return True
        _print_health_short(state)
        return True

    if key == "/store":
        value = arg.lower()
        if value in {"on", "1", "true", "yes"}:
            state.store_turn = True
        elif value in {"off", "0", "false", "no"}:
            state.store_turn = False
        else:
            print("Usage: /store on|off")
            return True
        _save_ui_state(state)
        print(f"store_turn: {'on' if state.store_turn else 'off'}")
        return True

    if not _ensure_connected_or_start(state):
        return True
    rc = _send_chat(state, raw, command_output=True)
    if rc == 0:
        if key == "/output":
            low = str(arg or "").strip().lower()
            if low in {"parameters on", "parameters off"}:
                _set_runtime_pref(state, "output_parameters", low.endswith("on"))
            elif low in {"summary on", "summary off"}:
                _set_runtime_pref(state, "output_summary", low.endswith("on"))
        elif key == "/mode_lock":
            low = str(arg or "").strip().lower()
            if low in {"on", "off"}:
                _set_runtime_pref(state, "mode_lock", low == "on")
        elif key == "/mode":
            low = str(arg or "").strip().lower()
            if low in {"auto", "off"}:
                _set_runtime_pref(state, "mode_lock", False)
            elif low and low not in {"current", "status"}:
                _set_runtime_pref(state, "active_mode", low)
                _set_runtime_pref(state, "mode_lock", True)
    return True


def _chat_once(state: ConsoleState, text: str) -> int:
    user_text = str(text or "").strip()
    if not user_text:
        return 0
    if _looks_like_shell_command(user_text):
        print("Это похоже на команду терминала. Вводи ее в PowerShell, не в `you>`.")
        return 1
    if not _ensure_connected_or_start(state):
        return 1
    _ensure_model_backend(state)
    return _send_chat(state, user_text)


def _chat_loop(state: ConsoleState) -> int:
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line.startswith("/"):
            if not _handle_command(state, line):
                return 0
            continue
        if _looks_like_shell_command(line):
            print("Это похоже на команду терминала. Вводи ее в PowerShell, не в `you>`.")
            continue
        if not _ensure_connected_or_start(state):
            continue
        _ensure_model_backend(state)
        _send_chat(state, line)


class _ConsoleChunkRenderer:
    _OUTPUT_RE = re.compile(r'"output"\s*:\s*"', flags=re.IGNORECASE)
    _HEX = set("0123456789abcdefABCDEF")

    def __init__(self):
        self._raw = ""
        self._mode = "unknown"  # unknown | plain | safety_json
        self._plain_pos = 0
        self._in_output = False
        self._output_done = False
        self._scan = 0
        self._escape = False
        self._unicode_digits: str | None = None

    def feed(self, piece: str) -> str:
        text = str(piece or "")
        if not text:
            return ""
        self._raw += text

        if self._mode == "plain":
            return self._drain_plain()
        if self._mode == "unknown":
            stripped = self._raw.lstrip()
            if stripped.startswith("{"):
                self._mode = "safety_json"
            elif stripped:
                self._mode = "plain"
                return self._drain_plain()
            return ""
        return self._drain_safety_json_output()

    def _drain_plain(self) -> str:
        out = self._raw[self._plain_pos :]
        self._plain_pos = len(self._raw)
        return out

    def _drain_safety_json_output(self) -> str:
        if self._output_done:
            return ""
        out: list[str] = []

        if not self._in_output:
            marker = self._OUTPUT_RE.search(self._raw)
            if marker is None:
                # Fallback: if stream does not look like safety envelope, show plain text.
                if len(self._raw) > 512:
                    self._mode = "plain"
                    return self._drain_plain()
                return ""
            self._in_output = True
            self._scan = marker.end()

        while self._scan < len(self._raw):
            ch = self._raw[self._scan]
            self._scan += 1

            if self._unicode_digits is not None:
                if ch in self._HEX:
                    self._unicode_digits += ch
                    if len(self._unicode_digits) == 4:
                        try:
                            out.append(chr(int(self._unicode_digits, 16)))
                        except Exception:
                            out.append("\\u" + self._unicode_digits)
                        self._unicode_digits = None
                    continue
                out.append("\\u" + self._unicode_digits)
                self._unicode_digits = None

            if self._escape:
                self._escape = False
                if ch == "u":
                    self._unicode_digits = ""
                    continue
                mapping = {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                }
                out.append(mapping.get(ch, ch))
                continue

            if ch == "\\":
                self._escape = True
                continue
            if ch == '"':
                self._output_done = True
                self._in_output = False
                break
            out.append(ch)

        return "".join(out)


class _StreamRealtimePrinter:
    """Прямой вывод стриминга без буферизации (как в нативном Ollama)."""
    
    def __init__(
        self,
        renderer: _ConsoleChunkRenderer,
        *,
        prefer_thinking_first: bool = True,
        show_thinking: bool = True,
        debug_memory: bool = False,
    ):
        self.renderer = renderer
        self.show_thinking = bool(show_thinking)
        self.debug_memory = bool(debug_memory)
        self.answer_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self._thinking_started = False
        self._io_lock = threading.RLock()
        self._hidden_hint_frames = (
            "\u0434\u0443\u043c\u0430\u0435\u0442.",
            "\u0434\u0443\u043c\u0430\u0435\u0442..",
            "\u0434\u0443\u043c\u0430\u0435\u0442...",
        )
        self._hidden_hint_index = 0
        self._hidden_hint_active = False
        self._hidden_hint_stop = threading.Event()
        self._hidden_hint_thread: threading.Thread | None = None
        self._hidden_hint_last_width = 0
        # Channel tracking for streaming output
        self._current_channel: str | None = None
        # Debug memory state
        self._memory_retrieval_shown = False

    def start_hidden_thinking_hint(self) -> None:
        if self.show_thinking:
            return
        with self._io_lock:
            if self._hidden_hint_active:
                return
            self._hidden_hint_active = True
            self._hidden_hint_stop = threading.Event()
            self._hidden_hint_index = 1
            self._render_hidden_hint_locked(self._hidden_hint_frames[0])
            self._hidden_hint_thread = threading.Thread(
                target=self._run_hidden_thinking_hint,
                name="mmis-thinking-indicator",
                daemon=True,
            )
            self._hidden_hint_thread.start()

    def stop_hidden_thinking_hint(self, *, clear_line: bool = True) -> None:
        thread: threading.Thread | None = None
        with self._io_lock:
            if not self._hidden_hint_active and self._current_channel != "thinking_hint":
                return
            self._hidden_hint_active = False
            self._hidden_hint_stop.set()
            thread = self._hidden_hint_thread
            self._hidden_hint_thread = None
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.25)
        with self._io_lock:
            if clear_line and self._current_channel == "thinking_hint":
                self._clear_hidden_hint_line_locked()

    def _run_hidden_thinking_hint(self) -> None:
        while not self._hidden_hint_stop.wait(0.35):
            with self._io_lock:
                if not self._hidden_hint_active:
                    break
                frame = self._hidden_hint_frames[self._hidden_hint_index]
                self._hidden_hint_index = (self._hidden_hint_index + 1) % len(self._hidden_hint_frames)
                self._render_hidden_hint_locked(frame)

    def _render_hidden_hint_locked(self, frame: str) -> None:
        line = f"assistant> {str(frame or '')}"
        width = max(self._hidden_hint_last_width, len(line))
        sys.stdout.write("\r" + line.ljust(width))
        sys.stdout.flush()
        self._hidden_hint_last_width = width
        self._printed_any = True
        self._current_channel = "thinking_hint"

    def _clear_hidden_hint_line_locked(self) -> None:
        width = max(0, int(self._hidden_hint_last_width))
        if width > 0:
            sys.stdout.write("\r" + (" " * width) + "\r")
            sys.stdout.flush()
        self._hidden_hint_last_width = 0
        self._current_channel = None
        self._printed_any = False

    def on_thinking(self, piece: str) -> None:
        """Вывод thinking чанка сразу без буферизации."""
        text = _sanitize_stream_text(piece)
        if not text:
            return
        self._thinking_started = True
        self.thinking_parts.append(text)
        if not self.show_thinking:
            self.start_hidden_thinking_hint()
            return
        with self._io_lock:
            # Thinking выводится в отдельной строке перед ответом
            sys.stdout.write(text)
            sys.stdout.flush()

    def on_answer(self, piece: str) -> None:
        """Вывод answer чанка сразу без буферизации (как в нативном Ollama)."""
        raw = str(piece or "")
        if not raw:
            return
        text = _sanitize_stream_text(raw)
        if not text:
            return

        # Выводим сразу без буферизации
        with self._io_lock:
            sys.stdout.write(text)
            sys.stdout.flush()
        self.answer_parts.append(text)

    def finalize(self) -> None:
        self.stop_hidden_thinking_hint(clear_line=True)

    def rendered_answer(self) -> str:
        return "".join(self.answer_parts)

    def rendered_thinking(self) -> str:
        return "".join(self.thinking_parts)

    def finalize_with_final(self, *, answer_final: str | None = None, thinking_final: str | None = None, debug_trace: dict | None = None) -> None:
        """Завершение стриминга с финальными данными."""
        self.stop_hidden_thinking_hint(clear_line=True)

        # Show debug memory info (after answer and thinking)
        if self.debug_memory and not self._memory_retrieval_shown:
            self._show_debug_memory(debug_trace)
            self._memory_retrieval_shown = True

    def _show_debug_memory(self, debug_trace: dict | None) -> None:
        """Показать debug memory информацию после ответа."""
        if not debug_trace:
            return
        
        memory_retrieval = debug_trace.get("memory_retrieval", {})
        if not memory_retrieval:
            print("\n[memory retrieval] no retrieval performed (gate not triggered)")
            sys.stdout.flush()
            return
        
        print("\n[memory retrieval]")
        sys.stdout.flush()
        
        # Retrieved memories
        selected = memory_retrieval.get("selected_total", 0)
        if selected > 0:
            print(f"  found: {selected} memories")
            sys.stdout.flush()
            
            # Facts
            facts = memory_retrieval.get("selected_facts", [])
            if facts:
                print(f"  facts: {len(facts)}")
                for fact in facts[:3]:
                    print(f"    - {fact.get('text', '')[:80]}")
                sys.stdout.flush()
            
            # Claims
            claims = memory_retrieval.get("selected_claims", [])
            if claims:
                print(f"  claims: {len(claims)}")
                for claim in claims[:3]:
                    print(f"    - {claim.get('text', '')[:80]}")
                sys.stdout.flush()
            
            # Messages
            messages = memory_retrieval.get("selected_messages", [])
            if messages:
                print(f"  messages: {len(messages)}")
                for msg in messages[:3]:
                    print(f"    - {msg.get('text', '')[:80]}")
                sys.stdout.flush()
            
            # Episodes
            episodes = memory_retrieval.get("selected_episodes", [])
            if episodes:
                print(f"  episodes: {len(episodes)}")
                for ep in episodes[:3]:
                    print(f"    - {ep.get('summary_short', '')[:80]}")
                sys.stdout.flush()
        
        # Confidence
        confidence = memory_retrieval.get("confidence", {})
        if confidence:
            top_score = confidence.get("top_selected_score", 0)
            print(f"  confidence: {top_score:.2f}")
            sys.stdout.flush()
        
        # Query info
        query = memory_retrieval.get("query", "")
        if query:
            print(f"  query: {query[:60]}")
            sys.stdout.flush()
        
        # Stage info
        stage = memory_retrieval.get("stage", "")
        if stage:
            print(f"  stage: {stage}")
            sys.stdout.flush()


def _sanitize_stream_text(piece: str) -> str:
    src = str(piece or "")
    if not src:
        return ""
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in src if (ch == "\n" or ch == "\t" or ord(ch) >= 32))


def _send_chat(state: ConsoleState, text: str, *, command_output: bool = False) -> int:
    """Отправка сообщения и прямой вывод ответа (как в нативном Ollama)."""
    try:
        reply, streamed_text, streamed_thinking = _stream_once(state, text, command_output=command_output)
        if _is_generation_fallback(reply.answer):
            # Проверяем, действительно ли backend недоступен
            backend_dead = False
            try:
                state.api.health()
            except Exception:
                backend_dead = True

            if backend_dead:
                print()
                print("LLM backend недоступен. Пробую автоматически переключиться на локальный Ollama и повторить запрос...")
                if _ensure_model_backend(state):
                    reply, streamed_text, streamed_thinking = _stream_once(state, text, command_output=command_output)

        if _is_generation_fallback(reply.answer):
            _print_backend_hint(state)

        # Новая строка после ответа
        print()
        if not bool(command_output):
            model = str(getattr(reply, "model", "") or "")
            if model:
                print(f"[model: {model}]")
        state.online = True
        return 0
    except ApiClientError as exc:
        state.online = False
        print()
        print(f"API error: {exc}")
        return 1

def _stream_once(state: ConsoleState, text: str, *, command_output: bool = False):
    """Стриминг запроса с прямым выводом в консоль."""
    renderer = _ConsoleChunkRenderer()
    prefer_thinking_first = bool(state.show_thinking and state.thinking_first)
    printer = _StreamRealtimePrinter(
        renderer,
        prefer_thinking_first=prefer_thinking_first,
        show_thinking=bool(state.show_thinking),
        debug_memory=bool(state.debug_memory),
    )
    if (not bool(command_output)) and (not bool(state.show_thinking)) and bool(state.think_enabled):
        printer.start_hidden_thinking_hint()
    
    # Печатаем префикс перед стримингом (как в нативном Ollama)
    if not command_output:
        sys.stdout.write("assistant> ")
        sys.stdout.flush()
    
    try:
        reply = state.api.stream_chat(
            text=text,
            store_turn=state.store_turn,
            think=bool(state.think_enabled),
            json_mode=state.json_mode_enabled,
            on_chunk=printer.on_answer,
            on_thinking_chunk=printer.on_thinking,
        )
    finally:
        printer.stop_hidden_thinking_hint(clear_line=True)
    printer.finalize_with_final(
        answer_final=reply.answer,
        thinking_final=reply.thinking,
        debug_trace=reply.debug_trace,
    )
    return reply, printer.rendered_answer(), printer.rendered_thinking()


def _print_stream_piece(piece: str, *, sink: list[str] | None = None, renderer: _ConsoleChunkRenderer | None = None) -> None:
    raw = str(piece or "")
    if not raw:
        return
    text = renderer.feed(raw) if renderer is not None else raw
    if not text:
        return
    if sink is not None:
        sink.append(text)
    sys.stdout.write(text)
    sys.stdout.flush()


def _ensure_connected(state: ConsoleState) -> bool:
    if state.online:
        return True
    try:
        payload = state.api.health()
        state.online = True
        state.think_enabled = bool(payload.get("thinking_enabled", True))
        state.json_mode_enabled = bool(payload.get("json_mode_enabled", False))
        return True
    except ApiClientError:
        state.online = False
        return False


def _ensure_connected_or_start(state: ConsoleState) -> bool:
    if _ensure_connected(state):
        return True
    if state.auto_start_api and _start_api_process(state) and _wait_for_api(state, timeout_s=18.0):
        return True
    print("API offline. Запусти: python api_main.py --mmis-tag mmis")
    return False


def _ensure_model_backend(state: ConsoleState) -> bool:
    if not _ensure_connected_or_start(state):
        return False
    try:
        payload = state.api.list_models()
        models = [str(x) for x in list(payload.get("models") or []) if str(x).strip()]
        if models:
            return True
    except ApiClientError:
        pass

    if not state.auto_start_ollama:
        return False
    if not _looks_like_ollama_runtime(state.api.get_runtime_model()):
        return False
    if _start_ollama_process(state):
        return _wait_for_models(state, timeout_s=14.0)
    return False


def _wait_for_models(state: ConsoleState, timeout_s: float = 12.0) -> bool:
    deadline = time.time() + max(2.0, float(timeout_s))
    while time.time() < deadline:
        try:
            payload = state.api.list_models()
            models = [str(x) for x in list(payload.get("models") or []) if str(x).strip()]
            if models:
                return True
        except ApiClientError:
            pass
        time.sleep(0.4)
    return False


def _start_ollama_process(state: ConsoleState) -> bool:
    proc = state.ollama_process
    if proc is not None and proc.poll() is None:
        return True
    try:
        state.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _stop_owned_ollama_process(state: ConsoleState) -> bool:
    proc = state.ollama_process
    if proc is None:
        return False
    if proc.poll() is not None:
        state.ollama_process = None
        return True

    try:
        _stop_runtime_ollama_model(state, check_api_health=False)
    except Exception:
        pass

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
            if int(result.returncode or 0) != 0:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                )
        else:
            proc.terminate()
            proc.wait(timeout=4.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    state.ollama_process = None
    return True


def _is_local_api_base_url(base_url: str) -> bool:
    raw = str(base_url or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    except Exception:
        return False
    host = str(parsed.hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _stop_runtime_ollama_model(state: ConsoleState, *, check_api_health: bool = True) -> bool:
    if not _is_local_api_base_url(getattr(state.api, "base_url", "")):
        return False

    if check_api_health and bool(state.online):
        try:
            state.api.health()
        except Exception:
            pass

    get_runtime_model = getattr(state.api, "get_runtime_model", None)
    if not callable(get_runtime_model):
        return False
    model = str(get_runtime_model() or "").strip()
    if not model:
        return False
    return _stop_ollama_models([model])


def _stop_memory_ollama_model() -> bool:
    try:
        payload = json.loads((Path(__file__).resolve().parent / "memory_core" / "config.json").read_text(encoding="utf-8-sig") or "{}")
    except Exception:
        return False

    profiles = payload.get("task_model_profiles")
    if not isinstance(profiles, dict):
        return False

    profile = profiles.get("memory_llm_process")
    if not isinstance(profile, dict):
        return False

    provider = str(profile.get("provider") or "").strip().lower()
    if provider and provider != "ollama":
        return False

    model = str(profile.get("model") or "").strip()
    if not model:
        return False
    return _stop_ollama_models([model])


def _stop_ollama_models(models: list[str]) -> bool:
    targets: list[str] = []
    seen: set[str] = set()
    for model in list(models or []):
        src = str(model or "").strip()
        if not _looks_like_ollama_runtime(src):
            continue
        for target in _resolve_ollama_stop_targets(src):
            normalized = str(target or "").strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            targets.append(str(target))
    if not targets:
        return False

    try:
        for _ in range(2):
            for target in targets:
                subprocess.run(
                    ["ollama", "stop", str(target)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=12.0,
                )
            if _wait_ollama_models_unloaded(targets, timeout_s=3.0):
                return True
        return False
    except Exception:
        return False


def _resolve_ollama_stop_targets(model: str) -> list[str]:
    target = str(model or "").strip()
    if not target:
        return []
    ps_rows = _ollama_cli_name_id_rows(["ollama", "ps"])
    if not ps_rows:
        return [target]

    target_low = target.lower()
    list_rows = _ollama_cli_name_id_rows(["ollama", "list"])
    target_ids = {str(row_id or "").strip().lower() for row_name, row_id in list_rows if str(row_name or "").strip().lower() == target_low}

    out: list[str] = []
    seen: set[str] = set()
    for row_name, row_id in ps_rows:
        name = str(row_name or "").strip()
        low = name.lower()
        row_id_low = str(row_id or "").strip().lower()
        if not name:
            continue
        if low == target_low or (target_ids and row_id_low in target_ids):
            if low not in seen:
                seen.add(low)
                out.append(name)
    if not out:
        out.append(target)
    return out


def _wait_ollama_models_unloaded(models: list[str], *, timeout_s: float = 3.0) -> bool:
    targets = {str(x or "").strip().lower() for x in list(models or []) if str(x or "").strip()}
    if not targets:
        return True
    deadline = time.time() + max(0.5, float(timeout_s))
    while time.time() < deadline:
        if not _ollama_ps_has_any(targets):
            return True
        time.sleep(0.25)
    return (not _ollama_ps_has_any(targets))


def _ollama_ps_has_any(targets_lower: set[str]) -> bool:
    targets = {str(x or "").strip().lower() for x in set(targets_lower or set()) if str(x or "").strip()}
    if not targets:
        return False
    rows = _ollama_cli_name_id_rows(["ollama", "ps"])
    for row_name, _ in rows:
        name = str(row_name or "").strip().lower()
        if name in targets:
            return True
    return False


def _ollama_cli_name_id_rows(command: list[str]) -> list[tuple[str, str]]:
    try:
        row = subprocess.run(
            list(command or []),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8.0,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    if int(row.returncode or 0) != 0:
        return []
    lines = [str(line or "").strip() for line in str(row.stdout or "").splitlines() if str(line or "").strip()]
    if len(lines) < 2:
        return []
    out: list[tuple[str, str]] = []
    for line in lines[1:]:
        parts = str(line).split()
        if len(parts) < 2:
            continue
        out.append((str(parts[0]), str(parts[1])))
    return out


def _looks_like_ollama_runtime(model: str) -> bool:
    src = str(model or "").strip().lower()
    if not src:
        return True
    if src.startswith("gpt-") or src.startswith("o"):
        return False
    return (":" in src) or ("qwen" in src) or ("llama" in src) or ("mistral" in src)


def _is_generation_fallback(answer: str) -> bool:
    src = str(answer or "").strip().lower()
    return ("i got stuck during generation" in src) or ("please try again" in src)


def _print_backend_hint(state: ConsoleState) -> None:
    runtime_model = state.api.get_runtime_model() or "вЂ”"
    models: list[str] = []
    try:
        payload = state.api.list_models()
        models = [str(x) for x in list(payload.get("models") or []) if str(x).strip()]
    except ApiClientError:
        pass

    print("Диагностика:")
    print(f"- runtime model: {runtime_model}")
    if models:
        print(f"- models available: {len(models)}")
        if runtime_model not in models:
            print("- runtime model не найдена в списке. Смени модель: /models -> /model <name>")
    else:
        print("- models available: 0")
        print("- Ollama не отдает список моделей. Проверь:")
        print("  1) ollama serve")
        print("  2) ollama list")
        print("  3) ollama pull qwen3:8b")


def _start_api_process(state: ConsoleState) -> bool:
    proc = state.api_process
    if proc is not None and proc.poll() is None:
        return True

    root = Path(__file__).resolve().parent
    api_main = root / "api_main.py"
    if not api_main.exists():
        print(f"api_main.py not found: {api_main}")
        return False

    clean_mmis_api_processes(tag=MMIS_API_TAG, root=root)

    try:
        # Запускаем с перенаправлением вывода в файл для отладки
        import tempfile
        log_file = Path(tempfile.gettempdir()) / f"mmis_api_{os.getpid()}.log"
        popen_kwargs = {
            "cwd": str(root),
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        with open(log_file, "w", encoding="utf-8") as f:
            state.api_process = subprocess.Popen(
                [sys.executable, str(api_main), "--mmis-tag", MMIS_API_TAG, "--quiet"],
                stdout=f,
                **popen_kwargs,
            )
        # Ждём немного чтобы проверить запуск
        time.sleep(2.0)
        if state.api_process.poll() is not None:
            # Процесс завершился, читаем лог
            with open(log_file, "r", encoding="utf-8") as f:
                error_log = f.read()
            print(f"API process failed to start. Log: {error_log[:500]}")
            state.api_process = None
            return False
        return True
    except Exception as exc:
        print(f"Cannot start API automatically: {exc}")
        state.api_process = None
        return False


def _wait_for_api(state: ConsoleState, timeout_s: float = 25.0) -> bool:
    """Ждём запуска API (увеличено до 25 сек для надёжности)."""
    deadline = time.time() + max(2.0, float(timeout_s))
    while time.time() < deadline:
        if _ensure_connected(state):
            return True
        proc = state.api_process
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(0.5)  # Увеличили интервал проверки
    return False


def _restart_api_process(state: ConsoleState) -> bool:
    _stop_api_process(state)
    state.online = False
    if not _start_api_process(state):
        return False
    return _wait_for_api(state, timeout_s=18.0)


def _graceful_stop_api_process(proc: subprocess.Popen | None) -> bool:
    if proc is None or proc.poll() is not None:
        return True

    if os.name == "nt":
        ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if ctrl_break is not None:
            try:
                proc.send_signal(ctrl_break)
                proc.wait(timeout=8.0)
                return True
            except Exception:
                pass

    try:
        proc.terminate()
        proc.wait(timeout=4.0)
        return True
    except Exception:
        return False


def _stop_api_process(state: ConsoleState) -> None:
    proc = state.api_process
    if proc is None:
        _stop_runtime_ollama_model(state)
        _stop_memory_ollama_model()
        _stop_owned_ollama_process(state)
        return
    if proc.poll() is not None:
        state.api_process = None
        _stop_runtime_ollama_model(state, check_api_health=False)
        _stop_memory_ollama_model()
        _stop_owned_ollama_process(state)
        return
    try:
        stopped_gracefully = _graceful_stop_api_process(proc)
        if not stopped_gracefully and proc.poll() is None:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    state.api_process = None
    _stop_runtime_ollama_model(state, check_api_health=False)
    _stop_memory_ollama_model()
    _stop_owned_ollama_process(state)


def _cleanup(state: ConsoleState) -> None:
    _stop_api_process(state)


def _restart_full_app(state: ConsoleState) -> bool:
    root = Path(__file__).resolve().parent
    _save_ui_state(state)
    _stop_api_process(state)
    clean_mmis_api_processes(tag=MMIS_API_TAG, root=root)
    state.online = False
    if not _start_api_process(state):
        return False
    return _wait_for_api(state, timeout_s=18.0)

def _looks_like_shell_command(text: str) -> bool:
    src = str(text or "").strip()
    if not src:
        return False
    return bool(
        re.match(r"^(python|py|powershell|cmd|\.\\|[A-Za-z]:\\).+", src, flags=re.I)
        or "ui_console.py" in src.lower()
    )


def main() -> int:
    _configure_stdout()
    args = _build_parser().parse_args()
    persisted_ui_state = _load_ui_state()
    settings_runtime = _load_console_runtime_from_settings_file()

    api_url = str(args.api_url).strip() or config.api_url
    persisted_runtime = dict(persisted_ui_state.get("runtime") or {}) if isinstance(persisted_ui_state.get("runtime"), dict) else {}
    if settings_runtime:
        persisted_runtime = {**persisted_runtime, **settings_runtime}
        persisted_ui_state["runtime"] = dict(persisted_runtime)
    model_name = str(args.model).strip()
    if not model_name:
        model_name = str(config.model_name or "").strip()
    timeout = float(args.timeout) if args.timeout is not None else float(config.console_timeout_sec)
    stream_timeout = float(args.stream_timeout) if args.stream_timeout is not None else float(config.console_stream_timeout_sec)

    if args.no_store:
        store_turn = False
    else:
        store_turn = config.console_store_turn
    
    if args.show_thinking:
        show_thinking = True
    elif args.hide_thinking:
        show_thinking = False
    else:
        show_thinking = config.console_show_thinking
    if bool(args.thinking_first):
        thinking_first = True
    elif bool(args.thinking_last):
        thinking_first = False
    else:
        thinking_first = bool(config.console_thinking_first)

    auto_start_api = False if args.no_auto_api else config.console_auto_start_api
    auto_start_ollama = False if args.no_auto_ollama else config.console_auto_start_ollama

    state = ConsoleState(
        api=ApiClient(
            base_url=api_url,
            timeout_sec=timeout,
            stream_timeout_sec=stream_timeout,
        ),
        store_turn=store_turn,
        show_thinking=show_thinking,
        thinking_first=thinking_first,
        auto_start_api=auto_start_api,
        auto_start_ollama=auto_start_ollama,
        prefs=dict(persisted_ui_state),
    )

    _print_header(state)

    # НЕ проверяем подключение при старте - API будет запущен когда понадобится
    # Это предотвращает запуск Memory LLM до появления поля ввода
    api_available = False  # _ensure_connected(state)
    
    if model_name:
        # Пробуем установить модель только если API уже запущен
        try:
            state.api.set_model(model_name)
            print(f"Startup model set: {state.api.get_runtime_model() or model_name}")
        except ApiClientError:
            pass  # API ещё не запущен, установим позже

    think_action = None
    if bool(args.think):
        think_action = True
    elif bool(args.nothink):
        think_action = False
    else:
        think_action = bool(config.thinking_enabled)

    if think_action is not None:
        # Применяем настройку только если API доступен
        try:
            state.think_enabled = bool(state.api.set_thinking_enabled(think_action))
        except ApiClientError:
            pass  # API ещё не запущен

    json_action = None
    if bool(args.json):
        json_action = True
    elif bool(args.nojson):
        json_action = False
    else:
        json_action = bool(config.json_mode_enabled)

    if json_action is not None:
        try:
            state.json_mode_enabled = bool(state.api.set_json_mode_enabled(json_action))
        except ApiClientError:
            pass  # API ещё не запущен

    # НЕ применяем web_mode и persisted settings при старте
    # if api_available:
    #     ...

    if state.think_enabled is not None:
        print(f"Thinking: {'on' if state.think_enabled else 'off'}")

    try:
        once = str(args.once).strip()
        if once:
            return _chat_once(state, once)
        return _chat_loop(state)
    finally:
        _cleanup(state)


if __name__ == "__main__":
    raise SystemExit(main())
