from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ui.api_client import ApiClient, ApiClientError
from config.settings import load_config


CONSOLE_BUILD_ID = "2026-02-28-r2"
config = load_config()

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
    print("Запусти вручную: python main.py --mode api")


def _print_help() -> None:
    print("/help                        show commands")
    print("/connect [url]               reconnect API (optional custom url)")
    print("/restartapi                  restart local API process")
    print("/models                      list available models")
    print("/model                       show current runtime model")
    print("/model <name>                set runtime model")
    print("/think                       enable thinking")
    print("/nothink                     disable thinking")
    print("/show-thinking               show thinking stream")
    print("/hide-thinking               hide thinking stream")
    print("/thinking-first              prefer thinking stream before answer")
    print("/thinking-last               prefer answer stream without waiting for thinking")
    print("/web                         enable web search")
    print("/no-web                      disable web search")
    print("/web-auto                    auto web search")
    print("/output status               show output format controls")
    print("/output parameters on|off")  
    print("/output summary on|off")     
    print("/mode <name>                 backend mode switch")
    print("/mode_lock on|off            backend mode lock")
    print("/brain_debug                 backend brain debug")
    print("/persona_debug               backend persona debug")
    print("/json                        enable JSON mode")
    print("/nojson                      disable JSON mode")
    print("/character ...               backend character command")
    print("/trait ...                   backend trait command")
    print("/health                      show API health")
    print("/store on|off                toggle store_turn")
    print("/exit or /quit               exit")


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
            print(f"Current model: {state.api.get_runtime_model() or '—'}")
            return True
        try:
            state.api.set_model(arg)
            print(f"Model switched to: {state.api.get_runtime_model() or arg}")
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
            print(f"Thinking: {'on' if actual else 'off'}")
        except ApiClientError as exc:
            state.online = False
            print(f"API error: {exc}")
        return True
    
    if key in {"/show-thinking", "/hide-thinking"}:
        target = (key == "/show-thinking")
        state.show_thinking = target
        print(f"Thinking display: {'on' if target else 'off'}")
        return True

    if key in {"/thinking-first", "/thinking-last"}:
        state.thinking_first = (key == "/thinking-first")
        print(f"Thinking-first: {'on' if state.thinking_first else 'off'}")
        return True
    
    if key in {"/web", "/no-web", "/web-auto"}:
        if not _ensure_connected_or_start(state):
            return True

        target = "on" if key == "/web" else ("off" if key == "/no-web" else "auto")
        try:
            actual = str(state.api.set_web_mode(target))
            state.web_mode = actual
            print(f"Web mode: {actual}")
            if arg :
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
        print(f"store_turn: {'on' if state.store_turn else 'off'}")
        return True

    if not _ensure_connected_or_start(state):
        return True
    _send_chat(state, raw)
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
    def __init__(
        self,
        renderer: _ConsoleChunkRenderer,
        *,
        prefer_thinking_first: bool = True,
        show_thinking: bool = True,
    ):
        self.renderer = renderer
        self.prefer_thinking_first = bool(prefer_thinking_first)
        self.show_thinking = bool(show_thinking)
        self.answer_parts: list[str] = []
        self.thinking_parts: list[str] = []
        self._pending_answer: list[str] = []
        self._thinking_started = False
        self._printed_any = False
        self._current_channel = ""
        # Track how many sanitized characters have been physically printed
        self._answer_printed_chars: int = 0
        self._thinking_printed_chars: int = 0
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
        self._current_channel = ""
        self._printed_any = False

    def on_thinking(self, piece: str) -> None:
        text = _sanitize_stream_text(piece)
        if not text:
            return
        self._thinking_started = True
        self.thinking_parts.append(text)
        if not self.show_thinking:
            self.start_hidden_thinking_hint()
            return
        with self._io_lock:
            self._start_channel("thinking")
            sys.stdout.write(text)
            sys.stdout.flush()

    def on_answer(self, piece: str) -> None:
        raw = str(piece or "")
        if not raw:
            return
        text = _sanitize_stream_text(raw)
        if not text:
            return
        
        # If this is the very first piece of the actual answer, strip leading whitespace/newlines
        # so it doesn't drop to a new line below "assistant> "
        if not self.answer_parts and not self._pending_answer:
            text = text.lstrip()
            if not text:
                return

        if self.prefer_thinking_first and not self._thinking_started:
            # If we prefer thinking first, we only buffer if thinking hasn't started
            # But we can't buffer forever. If we get a lot of answer and no thinking,
            # we should just release it.
            if len("".join(self._pending_answer)) + len(text) > 100:
                self._emit_answer("".join(self._pending_answer) + text)
                self._pending_answer = []
                self.prefer_thinking_first = False
            else:
                self._pending_answer.append(text)
            return

        if self._pending_answer:
            text = "".join(self._pending_answer) + text
            self._pending_answer = []
        self._emit_answer(text)

    def finalize(self) -> None:
        self.stop_hidden_thinking_hint(clear_line=True)
        if self._pending_answer:
            self._emit_answer("".join(self._pending_answer))
            self._pending_answer = []

    def rendered_answer(self) -> str:
        return "".join(self.answer_parts)

    def rendered_thinking(self) -> str:
        return "".join(self.thinking_parts)

    def finalize_with_final(self, *, answer_final: str | None = None, thinking_final: str | None = None) -> None:
        self.stop_hidden_thinking_hint(clear_line=True)
        # Flush any pending buffer first
        if self._pending_answer:
            self._emit_answer("".join(self._pending_answer))
            self._pending_answer = []

        # Append any tail characters not yet streamed
        if isinstance(answer_final, str) and answer_final:
            final_san = _sanitize_stream_text(answer_final).strip()
            rendered = self.rendered_answer().strip()
            if not rendered:
                # Nothing was streamed at all — print the full final answer
                if final_san:
                    self._emit_answer(final_san)
            elif final_san.endswith(rendered):
                pass  # streamed text is a suffix of final — nothing to add (edge case)
            elif final_san.startswith(rendered):
                # Normal case: final extends what was streamed
                tail = final_san[len(rendered):]
                if tail:
                    self._emit_answer(tail)
            else:
                # Out-of-sync: find how much of the END of rendered matches final
                # Try progressively shorter suffixes until we find an overlap
                tail = ""
                for cut in range(min(len(rendered), len(final_san)), 0, -1):
                    if final_san.startswith(rendered[-cut:]):
                        tail = final_san[cut:]
                        break
                if not tail and len(final_san) > len(rendered):
                    # Last resort: just emit the extra length at the end
                    tail = final_san[len(rendered):]
                if tail:
                    self._emit_answer(tail)

        # Append any thinking tail not yet streamed
        if self.show_thinking and isinstance(thinking_final, str) and thinking_final:
            final_t_san = _sanitize_stream_text(thinking_final).strip()
            rendered_t = self.rendered_thinking().strip()
            if not rendered_t:
                if final_t_san:
                    self._emit_thinking(final_t_san)
            elif final_t_san.startswith(rendered_t):
                tail_t = final_t_san[len(rendered_t):]
                if tail_t:
                    self._emit_thinking(tail_t)

    def _emit_thinking(self, text: str) -> None:
        if not text:
            return
        self._thinking_started = False
        self.thinking_parts.append(text)
        self._thinking_printed_chars += len(text)
        with self._io_lock:
            self._start_channel("thinking")
            sys.stdout.write(text)
            sys.stdout.flush()

    def _emit_answer(self, text: str) -> None:
        if not text:
            return
        self.stop_hidden_thinking_hint(clear_line=True)
        # Force strip leading whitespace on the very first text chunk
        if not self.answer_parts:
            text = text.lstrip()
            if not text:
                return
        with self._io_lock:
            self._start_channel("assistant")
            self.answer_parts.append(text)
            self._answer_printed_chars += len(text)
            sys.stdout.write(text)
            sys.stdout.flush()

    def _start_channel(self, channel: str) -> None:
        target = "thinking" if str(channel or "").lower() == "thinking" else "assistant"
        if self._current_channel == target:
            return
        if self._current_channel == "thinking_hint":
            self._clear_hidden_hint_line_locked()
        if self._printed_any:
            sys.stdout.write("\n")
        if target == "thinking":
            sys.stdout.write("[Thinking] ")
        else:
            sys.stdout.write("assistant> ")
        self._printed_any = True
        self._current_channel = target


def _sanitize_stream_text(piece: str) -> str:
    src = str(piece or "")
    if not src:
        return ""
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in src if (ch == "\n" or ch == "\t" or ord(ch) >= 32))


def _send_chat(state: ConsoleState, text: str) -> int:
    try:
        reply, streamed_text, streamed_thinking = _stream_once(state, text)
        if _is_generation_fallback(reply.answer):
            print()
            print("LLM backend недоступен. Пробую поднять Ollama и повторить запрос...")
            if _ensure_model_backend(state):
                reply, streamed_text, streamed_thinking = _stream_once(state, text)
        if _is_generation_fallback(reply.answer):
            _print_backend_hint(state)

        if not str(streamed_text or "").strip() and str(reply.answer or "").strip():
            print(f"assistant> {str(reply.answer or '')}", end="")
        print()
        model = str(getattr(reply, "model", "") or "")
        if model:
            print(f"[model: {model}]")
        if state.show_thinking:
            thinking = str(getattr(reply, "thinking", "") or "")
            if str(streamed_thinking or "").strip():
                pass
            elif thinking.strip():
                print("[thinking]")
                print(thinking.strip())
            elif state.think_enabled is not False:
                print("[thinking] —")
        state.online = True
        return 0
    except ApiClientError as exc:
        state.online = False
        print()
        print(f"API error: {exc}")
        return 1


def _stream_once(state: ConsoleState, text: str):
    renderer = _ConsoleChunkRenderer()
    prefer_thinking_first = bool(state.show_thinking and state.thinking_first)
    printer = _StreamRealtimePrinter(
        renderer,
        prefer_thinking_first=prefer_thinking_first,
        show_thinking=bool(state.show_thinking),
    )
    if not bool(state.show_thinking) and bool(state.think_enabled):
        printer.start_hidden_thinking_hint()
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
    printer.finalize_with_final(answer_final=reply.answer, thinking_final=reply.thinking)
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
    print("API offline. Запусти: python main.py --mode api")
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
    runtime_model = state.api.get_runtime_model() or "—"
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
    main_py = root / "main.py"
    if not main_py.exists():
        print(f"main.py not found: {main_py}")
        return False

    try:
        state.api_process = subprocess.Popen(
            [sys.executable, str(main_py), "--mode", "api"],
            cwd=str(root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:
        print(f"Cannot start API automatically: {exc}")
        state.api_process = None
        return False


def _wait_for_api(state: ConsoleState, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + max(2.0, float(timeout_s))
    while time.time() < deadline:
        if _ensure_connected(state):
            return True
        proc = state.api_process
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(0.35)
    return False


def _restart_api_process(state: ConsoleState) -> bool:
    _stop_api_process(state)
    state.online = False
    if not _start_api_process(state):
        return False
    return _wait_for_api(state, timeout_s=18.0)


def _stop_api_process(state: ConsoleState) -> None:
    proc = state.api_process
    if proc is None:
        return
    if proc.poll() is not None:
        state.api_process = None
        return
    try:
        proc.terminate()
        proc.wait(timeout=4.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    state.api_process = None


def _cleanup(state: ConsoleState) -> None:
    _stop_api_process(state)

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

    api_url = str(args.api_url).strip() or config.api_url
    model_name = str(args.model).strip() or config.console_model
    timeout = float(args.timeout) if args.timeout is not None else float(config.console_timeout_sec)
    stream_timeout = float(args.stream_timeout) if args.stream_timeout is not None else float(config.console_stream_timeout_sec)

    store_turn = False if args.no_store else config.console_store_turn
    
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
    )

    _print_header(state)

    if model_name:
        if _ensure_connected_or_start(state):
            try:
                state.api.set_model(model_name)
                print(f"Startup model set: {state.api.get_runtime_model() or model_name}")
            except ApiClientError as exc:
                state.online = False
                print(f"API error: {exc}")

    think_action = None
    if bool(args.think):
        think_action = True
    elif bool(args.nothink):
        think_action = False
    elif config.thinking_enabled is not None:
        think_action = config.thinking_enabled

    if think_action is not None:
        if _ensure_connected_or_start(state):
            try:
                state.think_enabled = bool(state.api.set_thinking_enabled(think_action))
            except ApiClientError as exc:
                state.online = False
                print(f"API error: {exc}")

    json_action = None
    if bool(args.json):
        json_action = True
    elif bool(args.nojson):
        json_action = False
    elif config.console_json_mode_enabled:
        json_action = True

    if json_action is not None:
        if _ensure_connected_or_start(state):
            try:
                state.json_mode_enabled = bool(state.api.set_json_mode_enabled(json_action))
            except ApiClientError as exc:
                state.online = False
                print(f"API error: {exc}")

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
