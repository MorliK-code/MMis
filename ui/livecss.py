from __future__ import annotations

from pathlib import Path


class LiveCss:
    """Loads a CSS template from disk and reloads it when the file changes.

    The template can use ``str.format`` placeholders:
    ``{font_size_px}``, ``{bubble_opacity}``, and ``{user_opacity}``.
    """

    def __init__(self, css_path: Path):
        self.css_path = Path(css_path)
        self._mtime_ns: int | None = None
        self._template: str | None = None
        self.load_if_changed()

    def load_if_changed(self) -> bool:
        """Reload CSS file if mtime changed. Returns True when reloaded."""
        if not self.css_path.exists():
            if self._template is None:
                return False
            self._template = None
            self._mtime_ns = None
            return True

        stat = self.css_path.stat()
        if self._mtime_ns == stat.st_mtime_ns:
            return False

        self._template = self.css_path.read_text(encoding="utf-8")
        self._mtime_ns = stat.st_mtime_ns
        return True

    def render(self, fallback_css: str, font_size_px: int, bubble_opacity: float) -> str:
        """Render active CSS, defaulting to generated fallback if no file exists."""
        self.load_if_changed()
        if not self._template:
            return fallback_css

        user_opacity = min(max(bubble_opacity + 0.03, 0.04), 0.35)
        try:
            return self._template.format(
                font_size_px=int(font_size_px),
                bubble_opacity=float(bubble_opacity),
                user_opacity=float(user_opacity),
            )
        except (KeyError, ValueError):
            return self._template