from __future__ import annotations

import ctypes
import platform
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config.settings import load_config
from modules.screen.ocr import OCRExtractor, OCRLine
from utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ScreenElement:
    text: str
    bbox: tuple[int, int, int, int]
    kind: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScreenContext:
    screen_type: str
    tags: list[str] = field(default_factory=list)
    text: str = ""
    ocr_lines: list[OCRLine] = field(default_factory=list)
    key_elements: list[ScreenElement] = field(default_factory=list)
    active_window_title: str = ""
    active_window_process: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_type": self.screen_type,
            "tags": list(self.tags),
            "text": self.text,
            "ocr_lines": [x.to_dict() for x in self.ocr_lines],
            "key_elements": [x.to_dict() for x in self.key_elements],
            "active_window_title": self.active_window_title,
            "active_window_process": self.active_window_process,
            "metadata": dict(self.metadata or {}),
        }


class ScreenAnalyzer:
    def __init__(self, ocr: OCRExtractor | None = None, capture_dir: str | Path | None = None):
        cfg = load_config()
        default_capture_dir = cfg.memory_dir / "screen_captures"
        self.capture_dir = Path(capture_dir).expanduser() if capture_dir is not None else default_capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.ocr = ocr or OCRExtractor()

    def capture(self, region: tuple[int, int, int, int] | None = None) -> str | None:
        try:
            from PIL import ImageGrab  # type: ignore
        except Exception:
            return None

        try:
            image = ImageGrab.grab(bbox=region)
            path = self.capture_dir / f"screen-{int(time.time() * 1000)}.png"
            image.save(str(path))
            return str(path)
        except Exception as exc:
            LOGGER.warning("screen capture failed: %s", exc)
            return None

    def analyze(self, image: str | Path | bytes | None) -> ScreenContext:
        lines = self.ocr.extract_text(image or "")
        text = self.ocr.merge_lines(lines)
        title, process = _active_window_info()

        screen_type, tags = _classify_screen(title=title, process=process, text=text)
        elements = _extract_elements(lines)

        if any("traceback" in line.text.lower() or "error" in line.text.lower() for line in lines):
            if "screen_error_dialog" not in tags:
                tags.append("screen_error_dialog")

        return ScreenContext(
            screen_type=screen_type,
            tags=sorted(set(tags)),
            text=text,
            ocr_lines=lines,
            key_elements=elements,
            active_window_title=title,
            active_window_process=process,
            metadata={
                "line_count": len(lines),
                "has_ocr": bool(lines),
            },
        )


def _active_window_info() -> tuple[str, str]:
    if platform.system().lower() != "windows":
        return "", ""

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", ""
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = str(buffer.value or "")
        return title, ""
    except Exception:
        return "", ""


def _classify_screen(title: str, process: str, text: str) -> tuple[str, list[str]]:
    src = " ".join([str(title or ""), str(process or ""), str(text or "")]).lower()

    rules: list[tuple[str, list[str], list[str]]] = [
        ("browser", ["chrome", "firefox", "edge", "safari", "http", "www", "tab"], ["screen_browser"]),
        ("ide", ["vscode", "visual studio", "pycharm", "traceback", ".py", "terminal"], ["screen_vscode"]),
        ("chat", ["chat", "telegram", "discord", "slack", "assistant"], ["screen_chat"]),
        ("video", ["youtube", "video", "pause", "play"], ["screen_video"]),
        ("error_dialog", ["error", "exception", "traceback", "failed"], ["screen_error_dialog"]),
    ]

    best_type = "unknown"
    best_score = 0
    tags: list[str] = ["screen_unknown"]

    for candidate, keywords, candidate_tags in rules:
        score = sum(1 for kw in keywords if kw in src)
        if score > best_score:
            best_score = score
            best_type = candidate
            tags = list(candidate_tags)

    if best_type == "unknown":
        return best_type, tags

    if "error" in src or "traceback" in src:
        tags.append("screen_error_hint")
    if "settings" in src:
        tags.append("screen_settings")
    return best_type, tags


def _extract_elements(lines: list[OCRLine]) -> list[ScreenElement]:
    out: list[ScreenElement] = []
    button_words = {"ok", "cancel", "save", "apply", "run", "send", "submit", "close", "retry"}
    for line in lines:
        low = line.text.lower()
        kind = "text"
        confidence = 0.4

        if any(word == low or re.search(rf"\b{re.escape(word)}\b", low) for word in button_words):
            kind = "button"
            confidence = max(confidence, 0.65)
        elif any(token in low for token in ("error", "exception", "traceback")):
            kind = "error"
            confidence = max(confidence, 0.8)
        elif any(token in low for token in ("http://", "https://", "www.")):
            kind = "link"
            confidence = max(confidence, 0.7)

        out.append(ScreenElement(text=line.text, bbox=line.bbox, kind=kind, confidence=confidence))

    return out[:80]


_DEFAULT_ANALYZER = ScreenAnalyzer()


def capture(region: tuple[int, int, int, int] | None = None) -> str | None:
    return _DEFAULT_ANALYZER.capture(region)


def analyze(image: str | Path | bytes | None) -> ScreenContext:
    return _DEFAULT_ANALYZER.analyze(image)


def analyze_screen(image: str | Path | bytes | None) -> dict[str, Any]:
    return _DEFAULT_ANALYZER.analyze(image).to_dict()
