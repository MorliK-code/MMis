"""Hot-reload helper for chat CSS overrides."""

from __future__ import annotations

from pathlib import Path


class LiveCss:
    """Loads a CSS template from disk and reloads it when the file changes.

    The template can use plain token placeholders:
    ``{font_size_px}``, ``{bubble_opacity}``, and ``{user_opacity}``.
    We intentionally avoid ``str.format`` so regular CSS braces do not need escaping.
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
        rendered = self._template
        rendered = rendered.replace("{font_size_px}", str(int(font_size_px)))
        rendered = rendered.replace("{bubble_opacity}", f"{float(bubble_opacity):.3f}")
        rendered = rendered.replace("{user_opacity}", f"{float(user_opacity):.3f}")

        # QTextBrowser reliably applies styles when they are inside <style>...</style>.
        if "<style" not in rendered.lower():
            rendered = f"<style>\n{rendered}\n</style>"

        return rendered