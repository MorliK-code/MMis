from __future__ import annotations

import re
import webbrowser
from dataclasses import asdict, dataclass, field
from typing import Callable
from urllib.parse import urlparse

from modules.automation.os_actions import ActionResult, OSActions


@dataclass(frozen=True)
class BrowserConfig:
    read_only: bool = True
    allow_domains: list[str] = field(default_factory=list)
    require_confirmation_for_sensitive: bool = True


class BrowserController:
    """Browser actions adapter. No planning logic, only execution."""

    def __init__(
        self,
        *,
        os_actions: OSActions | None = None,
        config: BrowserConfig | None = None,
        confirm_callback: Callable[[str], bool] | None = None,
    ):
        self.os_actions = os_actions or OSActions()
        self.config = config or BrowserConfig()
        self.confirm_callback = confirm_callback

    def open_url(self, url: str) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        raw = str(url or "").strip()
        if not raw:
            return _res(False, "open_url", started, error="url is empty")

        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            return _res(False, "open_url", started, error="only http/https are allowed", data={"url": raw})

        if not self._is_allowed_domain(parsed.netloc):
            return _res(False, "open_url", started, error="domain is not allowed", data={"url": raw})

        sensitive = self._is_sensitive_url(raw)
        if sensitive and self.config.require_confirmation_for_sensitive:
            ok = bool(self.confirm_callback(raw)) if callable(self.confirm_callback) else False
            if not ok:
                return _res(
                    False,
                    "open_url",
                    started,
                    error="confirmation required for sensitive URL",
                    data={"url": raw},
                    requires_confirmation=True,
                )

        try:
            opened = webbrowser.open(raw, new=2)
            return _res(opened, "open_url", started, data={"url": raw})
        except Exception as exc:
            return _res(False, "open_url", started, error=str(exc), data={"url": raw})

    def search_in_page(self, text: str) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        payload = str(text or "").strip()
        if not payload:
            return _res(False, "search_in_page", started, error="text is empty")

        first = self.os_actions.hotkey("ctrl", "f")
        if not first.ok:
            return _res(False, "search_in_page", started, error=first.error or "hotkey failed", data={"text": payload})
        second = self.os_actions.type_text(payload)
        if not second.ok:
            return _res(False, "search_in_page", started, error=second.error or "type failed", data={"text": payload})
        return _res(True, "search_in_page", started, data={"text": payload})

    def click(self, selector_or_coords) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        if self.config.read_only:
            return _res(False, "click", started, error="read_only mode blocks click")
        result = self.os_actions.click(selector_or_coords)
        return _res(result.ok, "click", started, error=result.error, data=result.data)

    def type(self, text: str) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        if self.config.read_only:
            return _res(False, "type", started, error="read_only mode blocks typing")
        result = self.os_actions.type_text(text)
        return _res(result.ok, "type", started, error=result.error, data=result.data)

    def press(self, keys: str | list[str] | tuple[str, ...]) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        if self.config.read_only:
            return _res(False, "press", started, error="read_only mode blocks key press")
        result = self.os_actions.press(keys)
        return _res(result.ok, "press", started, error=result.error, data=result.data)

    def scroll(self, amount: int) -> ActionResult:
        from time import perf_counter

        started = perf_counter()
        if self.config.read_only:
            return _res(False, "scroll", started, error="read_only mode blocks scrolling")
        result = self.os_actions.scroll(amount)
        return _res(result.ok, "scroll", started, error=result.error, data=result.data)

    def _is_allowed_domain(self, netloc: str) -> bool:
        allow = [x.strip().lower() for x in list(self.config.allow_domains or []) if str(x).strip()]
        if not allow:
            return True
        host = str(netloc or "").lower().split(":")[0]
        return any(host == domain or host.endswith(f".{domain}") for domain in allow)

    @staticmethod
    def _is_sensitive_url(url: str) -> bool:
        src = str(url or "").lower()
        return bool(
            re.search(
                r"(login|signin|checkout|payment|purchase|bank|wallet|delete|account/settings)",
                src,
                flags=re.IGNORECASE,
            )
        )


def open_url(url: str) -> ActionResult:
    return _DEFAULT_BROWSER.open_url(url)


def _res(
    ok: bool,
    action: str,
    started: float,
    *,
    data: dict | None = None,
    error: str = "",
    requires_confirmation: bool = False,
) -> ActionResult:
    from time import perf_counter

    return ActionResult(
        ok=bool(ok),
        action=action,
        data=dict(data or {}),
        error=str(error or ""),
        duration_ms=(perf_counter() - started) * 1000.0,
        requires_confirmation=bool(requires_confirmation),
    )


_DEFAULT_BROWSER = BrowserController()
