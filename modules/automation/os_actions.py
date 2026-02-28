from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class OSActionConfig:
    safe_root: str | Path | None = None
    max_read_kb: int = 512
    max_write_kb: int = 512
    command_timeout_s: int = 30
    allow_shell: bool = True
    allow_input: bool = False
    allow_outside_read: bool = True
    allow_outside_write: bool = False


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    duration_ms: float = 0.0
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OSActions:
    def __init__(self, config: OSActionConfig | None = None):
        cfg = config or OSActionConfig()
        safe_root = Path(cfg.safe_root).expanduser().resolve() if cfg.safe_root else Path.cwd().resolve()
        self.config = OSActionConfig(
            safe_root=safe_root,
            max_read_kb=max(1, int(cfg.max_read_kb)),
            max_write_kb=max(1, int(cfg.max_write_kb)),
            command_timeout_s=max(1, int(cfg.command_timeout_s)),
            allow_shell=bool(cfg.allow_shell),
            allow_input=bool(cfg.allow_input),
            allow_outside_read=bool(cfg.allow_outside_read),
            allow_outside_write=bool(cfg.allow_outside_write),
        )

    def open_app(self, path_or_name: str) -> ActionResult:
        started = time.perf_counter()
        target = str(path_or_name or "").strip()
        if not target:
            return _result(False, "open_app", started, error="path_or_name is empty")

        try:
            if os.name == "nt" and Path(target).exists():
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen([target], shell=False)
            return _result(True, "open_app", started, data={"target": target})
        except Exception as exc:
            return _result(False, "open_app", started, error=str(exc), data={"target": target})

    def focus_window(self, title: str = "", process: str = "") -> ActionResult:
        started = time.perf_counter()
        title = str(title or "").strip()
        process = str(process or "").strip()
        if not title and not process:
            return _result(False, "focus_window", started, error="title or process is required")

        if os.name != "nt":
            return _result(False, "focus_window", started, error="focus_window implemented only on Windows")

        target = title or process
        escaped = target.replace("'", "''")
        script = f"(New-Object -ComObject WScript.Shell).AppActivate('{escaped}')"
        try:
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_s,
                check=False,
            )
            ok = cp.returncode == 0
            return _result(
                ok,
                "focus_window",
                started,
                stdout=cp.stdout,
                stderr=cp.stderr,
                data={"target": target, "returncode": cp.returncode},
                error="" if ok else "focus failed",
            )
        except Exception as exc:
            return _result(False, "focus_window", started, error=str(exc), data={"target": target})

    def run_command(self, cmd: str, *, capture: bool = True, cwd: str | Path | None = None, timeout_s: int | None = None) -> ActionResult:
        started = time.perf_counter()
        command = str(cmd or "").strip()
        if not command:
            return _result(False, "run_command", started, error="command is empty")
        if not self.config.allow_shell:
            return _result(False, "run_command", started, error="shell commands are disabled")

        timeout = int(timeout_s or self.config.command_timeout_s)
        run_cwd = str(Path(cwd).expanduser()) if cwd is not None else None
        try:
            cp = subprocess.run(
                command,
                shell=True,
                cwd=run_cwd,
                text=True,
                capture_output=bool(capture),
                timeout=max(1, timeout),
                check=False,
            )
            ok = cp.returncode == 0
            return _result(
                ok,
                "run_command",
                started,
                stdout=cp.stdout or "",
                stderr=cp.stderr or "",
                data={"returncode": cp.returncode, "cwd": run_cwd or ""},
                error="" if ok else f"command failed with code {cp.returncode}",
            )
        except subprocess.TimeoutExpired as exc:
            return _result(False, "run_command", started, error=f"timeout after {timeout}s", stderr=str(exc))
        except Exception as exc:
            return _result(False, "run_command", started, error=str(exc))

    def read_file(self, path: str | Path, *, max_kb: int | None = None) -> ActionResult:
        started = time.perf_counter()
        try:
            resolved = _resolve_path(path)
        except Exception as exc:
            return _result(False, "read_file", started, error=str(exc))

        if not resolved.exists() or not resolved.is_file():
            return _result(False, "read_file", started, error=f"file not found: {resolved}")
        if not self.config.allow_outside_read and not _is_under_root(resolved, Path(self.config.safe_root)):
            return _result(False, "read_file", started, error="path outside safe_root")

        limit_kb = int(max_kb or self.config.max_read_kb)
        limit_bytes = max(1, limit_kb) * 1024
        size = resolved.stat().st_size
        if size > limit_bytes:
            return _result(False, "read_file", started, error=f"file too large ({size} bytes > {limit_bytes} bytes)")

        try:
            content = resolved.read_text(encoding="utf-8-sig", errors="ignore")
            return _result(True, "read_file", started, data={"path": str(resolved), "content": content, "size": size})
        except Exception as exc:
            return _result(False, "read_file", started, error=str(exc), data={"path": str(resolved)})

    def write_file(self, path: str | Path, content: str) -> ActionResult:
        started = time.perf_counter()
        try:
            resolved = _resolve_path(path)
        except Exception as exc:
            return _result(False, "write_file", started, error=str(exc))

        if not self.config.allow_outside_write and not _is_under_root(resolved, Path(self.config.safe_root)):
            return _result(False, "write_file", started, error="path outside safe_root")

        payload = str(content or "")
        max_bytes = self.config.max_write_kb * 1024
        if len(payload.encode("utf-8")) > max_bytes:
            return _result(False, "write_file", started, error=f"content too large (> {max_bytes} bytes)")

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(payload, encoding="utf-8")
            return _result(True, "write_file", started, data={"path": str(resolved), "bytes": len(payload.encode('utf-8'))})
        except Exception as exc:
            return _result(False, "write_file", started, error=str(exc), data={"path": str(resolved)})

    def hotkey(self, *keys: str) -> ActionResult:
        started = time.perf_counter()
        if not self.config.allow_input:
            return _result(False, "hotkey", started, error="input actions are disabled")
        if not keys:
            return _result(False, "hotkey", started, error="no keys provided")

        try:
            import pyautogui  # type: ignore

            pyautogui.hotkey(*[str(k) for k in keys if str(k).strip()])
            return _result(True, "hotkey", started, data={"keys": list(keys)})
        except Exception as exc:
            return _result(False, "hotkey", started, error=str(exc), data={"keys": list(keys)})

    def press(self, keys: str | list[str] | tuple[str, ...]) -> ActionResult:
        if isinstance(keys, str):
            return self.hotkey(keys)
        return self.hotkey(*[str(k) for k in list(keys or [])])

    def type_text(self, text: str) -> ActionResult:
        started = time.perf_counter()
        if not self.config.allow_input:
            return _result(False, "type_text", started, error="input actions are disabled")

        payload = str(text or "")
        if not payload:
            return _result(False, "type_text", started, error="text is empty")

        try:
            import pyautogui  # type: ignore

            pyautogui.typewrite(payload, interval=0.0)
            return _result(True, "type_text", started, data={"chars": len(payload)})
        except Exception as exc:
            return _result(False, "type_text", started, error=str(exc))

    def click(self, selector_or_coords: Any) -> ActionResult:
        started = time.perf_counter()
        if not self.config.allow_input:
            return _result(False, "click", started, error="input actions are disabled")

        x, y = _coords(selector_or_coords)
        if x is None or y is None:
            return _result(False, "click", started, error="coords not provided")

        try:
            import pyautogui  # type: ignore

            pyautogui.click(x=int(x), y=int(y))
            return _result(True, "click", started, data={"x": int(x), "y": int(y)})
        except Exception as exc:
            return _result(False, "click", started, error=str(exc), data={"x": x, "y": y})

    def scroll(self, amount: int) -> ActionResult:
        started = time.perf_counter()
        if not self.config.allow_input:
            return _result(False, "scroll", started, error="input actions are disabled")

        try:
            import pyautogui  # type: ignore

            pyautogui.scroll(int(amount))
            return _result(True, "scroll", started, data={"amount": int(amount)})
        except Exception as exc:
            return _result(False, "scroll", started, error=str(exc), data={"amount": int(amount)})


def run_os_action(name: str, **kwargs) -> dict:
    actions = _DEFAULT_ACTIONS
    action = str(name or "").strip().lower()

    mapping = {
        "open_app": lambda: actions.open_app(kwargs.get("path") or kwargs.get("name") or ""),
        "focus_window": lambda: actions.focus_window(kwargs.get("title", ""), kwargs.get("process", "")),
        "run_command": lambda: actions.run_command(
            kwargs.get("cmd", ""),
            capture=bool(kwargs.get("capture", True)),
            cwd=kwargs.get("cwd"),
            timeout_s=kwargs.get("timeout_s"),
        ),
        "read_file": lambda: actions.read_file(kwargs.get("path", ""), max_kb=kwargs.get("max_kb")),
        "write_file": lambda: actions.write_file(kwargs.get("path", ""), kwargs.get("content", "")),
        "hotkey": lambda: actions.hotkey(*list(kwargs.get("keys") or [])),
        "type_text": lambda: actions.type_text(kwargs.get("text", "")),
        "click": lambda: actions.click(kwargs.get("selector_or_coords")),
        "press": lambda: actions.press(kwargs.get("keys") or kwargs.get("key") or ""),
        "scroll": lambda: actions.scroll(int(kwargs.get("amount", 0))),
    }

    handler = mapping.get(action)
    if handler is None:
        return _result(False, "run_os_action", time.perf_counter(), error=f"unknown action: {action}").to_dict()

    try:
        return handler().to_dict()
    except Exception as exc:
        return _result(False, "run_os_action", time.perf_counter(), error=str(exc), data={"action": action}).to_dict()


def _coords(value: Any) -> tuple[int | None, int | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return int(value["x"]), int(value["y"])
    return None, None


def _resolve_path(path: str | Path) -> Path:
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path is empty")
    return Path(raw).expanduser().resolve()


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _result(
    ok: bool,
    action: str,
    started: float,
    *,
    data: dict[str, Any] | None = None,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    requires_confirmation: bool = False,
) -> ActionResult:
    return ActionResult(
        ok=bool(ok),
        action=str(action),
        data=dict(data or {}),
        stdout=str(stdout or ""),
        stderr=str(stderr or ""),
        error=str(error or ""),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        requires_confirmation=bool(requires_confirmation),
    )


_DEFAULT_ACTIONS = OSActions()
