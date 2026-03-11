"""Hot-reload helper for chat CSS overrides."""

from __future__ import annotations

from pathlib import Path
import re


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
        text = self.css_path.read_text(encoding="utf-8")
        if self._mtime_ns == stat.st_mtime_ns and self._template == text:
            return False

        self._template = text
        self._mtime_ns = stat.st_mtime_ns
        return True

    @staticmethod
    def _expand_css_vars(css: str) -> str:
        """Expand :root custom properties into plain CSS for QTextBrowser."""
        root_blocks = re.findall(r":root\s*\{(.*?)\}", css, flags=re.DOTALL | re.IGNORECASE)
        if not root_blocks:
            return css

        vars_map: dict[str, str] = {}
        for body in root_blocks:
            for name, value in re.findall(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);", body):
                vars_map[name.strip()] = value.strip()

        if not vars_map:
            return css

        def replace_var(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            return vars_map.get(key, match.group(0))

        expanded = re.sub(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)", replace_var, css)
        # Remove custom-property blocks because QTextBrowser can reject them.
        expanded = re.sub(r":root\s*\{.*?\}", "", expanded, flags=re.DOTALL | re.IGNORECASE)
        return expanded

    @staticmethod
    def _parse_css_vars(css: str) -> dict[str, str]:
        """Parse CSS custom properties from :root blocks."""
        root_blocks = re.findall(r":root\s*\{(.*?)\}", css, flags=re.DOTALL | re.IGNORECASE)
        vars_map: dict[str, str] = {}
        for body in root_blocks:
            for name, value in re.findall(r"(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);", body):
                vars_map[name.strip()] = value.strip()
        return vars_map

    def get_vars(self, font_size_px: int, bubble_opacity: float) -> dict[str, str]:
        """Return resolved variables from active stylesheet template."""
        self.load_if_changed()
        if not self._template:
            return {}

        user_opacity = min(max(bubble_opacity + 0.03, 0.04), 0.35)
        css = self._template
        css = css.replace("{font_size_px}", str(int(font_size_px)))
        css = css.replace("{bubble_opacity}", f"{float(bubble_opacity):.3f}")
        css = css.replace("{user_opacity}", f"{float(user_opacity):.3f}")
        return self._parse_css_vars(css)

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
        rendered = self._expand_css_vars(rendered)

        # QTextBrowser reliably applies styles when they are inside <style>...</style>.
        if "<style" not in rendered.lower():
            rendered = f"<style>\n{rendered}\n</style>"

        return rendered
