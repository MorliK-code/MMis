"""HTTP client used by desktop UI to work with MMis API."""

from __future__ import annotations

import json
import os
import socket
import base64
from pathlib import Path
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import logging

from ui.client_config_store import get_api_access_key, get_selected_base_url
from ui.auth_client_store import get_auth_token, clear_auth_state

LOGGER = logging.getLogger(__name__)


class ApiClientError(RuntimeError):
    pass


@dataclass
class ApiReply:
    answer: str
    thinking: str
    thinking_generated: bool
    stats: dict
    model: str
    parameters: dict | None = None
    summary: str | None = None
    debug_trace: dict | None = None
    memory_debug_snapshot: dict | None = None


class ApiClient:
    def _normalize_url(self, url: str) -> str:
        if not url:
            return ""
        url = url.strip().rstrip("/")
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "http://" + url
        return url

    def __init__(self, base_url: str | None = None, timeout_sec: float = 5.0, stream_timeout_sec: float = 600.0):
        selected_url = get_selected_base_url()
        self.base_url = self._normalize_url(base_url or selected_url or "http://127.0.0.1:8027")
        self.timeout_sec = float(timeout_sec)
        self.stream_timeout_sec = float(stream_timeout_sec)
        self._runtime_model_cache = ""

    def set_base_url(self, base_url: str) -> None:
        """Updates the base URL at runtime."""
        self.base_url = self._normalize_url(base_url)

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _account_headers(self) -> dict[str, str]:
        account_id = str(os.environ.get("MMIS_ACTIVE_ACCOUNT_ID") or "").strip()
        if not account_id:
            try:
                from ui.auth_client_store import load_auth_state

                account_id = str(load_auth_state().get("account_id") or "").strip()
            except Exception:
                account_id = ""
        return {"X-MMis-Account-Id": account_id} if account_id else {}

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not text:
            return {}
        return json.loads(text)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout: float | None = None,
        *,
        include_api_access_key: bool = True,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        headers.update(self._account_headers())
        api_access_key = get_api_access_key() if include_api_access_key else ""
        if api_access_key:
            headers["X-MMis-Access-Key"] = api_access_key
        token = get_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib_request.Request(self._url(path), data=data, headers=headers, method=method.upper())
        LOGGER.debug("ui_api_request method=%s path=%s has_payload=%s", method, path, bool(payload))

        try:
            with urllib_request.urlopen(req, timeout=timeout or self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                out = self._parse_json(raw)
                LOGGER.debug("ui_api_response method=%s path=%s status=ok", method, path)
                return out
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            detail = raw
            try:
                payload = self._parse_json(raw)
                detail = str(payload.get("detail") or raw or exc)
            except Exception:
                detail = raw or str(exc)
            LOGGER.warning("ui api http error method=%s path=%s code=%s detail=%s", method, path, exc.code, detail)
            raise ApiClientError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            LOGGER.warning("ui api connection error method=%s path=%s url=%s reason=%s", method, path, self._url(path), exc.reason)
            raise ApiClientError(f"API connection failed ({self._url(path)}): {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            LOGGER.debug("ui api timeout method=%s path=%s timeout=%s", method, path, timeout or self.timeout_sec)
            raise ApiClientError(f"API request timed out: {path}") from exc
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("ui api unexpected error method=%s path=%s", method, path)
            raise ApiClientError(str(exc)) from exc


    def health(self, timeout: float | None = None) -> dict:
        payload = self._request_json("GET", "/health", timeout=timeout)
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return payload

    def ping(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/ping", timeout=timeout or 1.2)

    def auth_login(self, login: str, password: str) -> dict:
        payload = self._request_json(
            "POST",
            "/auth/login",
            {"login": str(login or ""), "password": str(password or "")},
            timeout=5.0,
        )
        return payload

    def auth_register(self, login: str, password: str, display_name: str = "") -> dict:
        payload = self._request_json(
            "POST",
            "/auth/register",
            {
                "login": str(login or ""),
                "password": str(password or ""),
                "display_name": str(display_name or ""),
            },
            timeout=5.0,
        )
        return payload

    def auth_me(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/auth/me", timeout=timeout or 3.0)

    def auth_logout(self) -> None:
        try:
            self._request_json("POST", "/auth/logout", timeout=3.0)
        finally:
            clear_auth_state(forget_current=True)

    def get_ui_chat_sessions(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/ui/chat-sessions", timeout=timeout or 3.0)

    def put_ui_chat_sessions(self, payload: dict, timeout: float | None = None) -> dict:
        return self._request_json("PUT", "/ui/chat-sessions", dict(payload or {}), timeout=timeout or 5.0)

    def transcribe_voice_file(self, path: str | os.PathLike, timeout: float | None = None) -> dict:
        audio_path = Path(path).expanduser()
        data = audio_path.read_bytes()
        payload = {
            "filename": audio_path.name,
            "audio_base64": base64.b64encode(data).decode("ascii"),
        }
        return self._request_json("POST", "/voice/transcribe", payload, timeout=timeout or 180.0)

    def admin_list_api_access_keys(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/auth/api-access-keys", timeout=timeout or 5.0, include_api_access_key=False)

    def admin_create_api_access_key(
        self,
        *,
        login: str,
        key: str = "",
        key_hash: str = "",
        label: str = "ui",
        replace: bool = False,
        timeout: float | None = None,
    ) -> dict:
        return self._request_json(
            "POST",
            "/auth/api-access-keys",
            {
                "login": str(login or ""),
                "key": str(key or ""),
                "key_hash": str(key_hash or ""),
                "label": str(label or "ui"),
                "replace": bool(replace),
            },
            timeout=timeout or 5.0,
            include_api_access_key=False,
        )

    def admin_set_api_access_key_enabled(self, key_hash: str, enabled: bool, timeout: float | None = None) -> dict:
        return self._request_json(
            "PATCH",
            f"/auth/api-access-keys/{str(key_hash or '').strip()}",
            {"enabled": bool(enabled)},
            timeout=timeout or 5.0,
            include_api_access_key=False,
        )

    def admin_delete_api_access_key(self, key_hash: str, timeout: float | None = None) -> dict:
        return self._request_json(
            "DELETE",
            f"/auth/api-access-keys/{str(key_hash or '').strip()}",
            timeout=timeout or 5.0,
            include_api_access_key=False,
        )

    def list_models(self, timeout: float | None = None) -> dict:
        payload = self._request_json("GET", "/models", timeout=timeout)
        self._runtime_model_cache = str(payload.get("runtime_model") or self._runtime_model_cache)
        return payload

    def get_runtime_model(self) -> str:
        return str(self._runtime_model_cache or "")

    def set_model(self, model_name: str) -> bool:
        payload = self._request_json("POST", "/models", {"model": str(model_name or "")})
        self._runtime_model_cache = str(payload.get("runtime_model") or self._runtime_model_cache)
        return True

    def set_thinking_enabled(self, enabled: bool) -> bool:
        payload = self._request_json("POST", "/thinking", {"enabled": bool(enabled)})
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return bool(payload.get("thinking_enabled", enabled))

    def set_verbose_enabled(self, enabled: bool) -> bool:
        payload = self._request_json("POST", "/verbose", {"enabled": bool(enabled)})
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return bool(payload.get("verbose_enabled", enabled))

    def set_web_mode(self, mode: str) -> str:
        payload = self._request_json("POST", "/web-mode", {"mode": str(mode or "")})
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return str(payload.get("web_mode") or mode)

    def set_json_mode_enabled(self, enabled: bool) -> bool:
        payload = self._request_json("POST", "/json-mode", {"enabled": bool(enabled)})
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return bool(payload.get("json_mode_enabled", enabled))

    def get_memory_inspector(
        self,
        *,
        conversation_id: str = "",
        limit: int = 80,
        include_store: bool = True,
    ) -> dict:
        query = urllib_parse.urlencode(
            {
                "conversation_id": str(conversation_id or ""),
                "limit": max(1, int(limit)),
                "include_store": "true" if include_store else "false",
            }
        )
        path = "/debug/memory-inspector"
        if query:
            path = f"{path}?{query}"
        return self._request_json("GET", path)

    def list_characters(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/characters", timeout=timeout or 3.0)

    def get_active_character(self, timeout: float | None = None) -> dict:
        return self._request_json("GET", "/characters/active", timeout=timeout or 3.0)

    def get_character(self, character_id: str, timeout: float | None = None) -> dict:
        return self._request_json("GET", f"/characters/{character_id}", timeout=timeout or 3.0)

    def set_active_character(self, character_id: str, timeout: float | None = None) -> dict:
        return self._request_json("POST", "/characters/active", {"id": str(character_id)}, timeout=timeout or 3.0)

    def create_character(
        self,
        character_id: str,
        name: str,
        *,
        llm_profile: str | None = None,
        default_mood: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        payload = {
            "id": str(character_id or "").strip().lower(),
            "name": str(name or "").strip(),
            "llm_profile": str(llm_profile or "BALANCED").strip().upper(),
            "default_mood": str(default_mood or "thoughtful").strip(),
        }
        return self._request_json("POST", "/characters", payload, timeout=timeout or 5.0)

    def update_character(self, character_id: str, updates: dict, timeout: float | None = None) -> dict:
        return self._request_json("PUT", f"/characters/{character_id}", dict(updates or {}), timeout=timeout or 5.0)

    def delete_character(self, character_id: str, timeout: float | None = None) -> dict:
        return self._request_json("DELETE", f"/characters/{character_id}", timeout=timeout or 5.0)

    def register_feedback(self, user_text: str, assistant_text: str, feedback: int, penalty: float = 0.2) -> None:
        self._request_json(
            "POST",
            "/feedback",
            {
                "user_text": str(user_text or ""),
                "assistant_text": str(assistant_text or ""),
                "feedback": int(feedback),
                "penalty": float(penalty),
            },
        )

    def stream_chat(
        self,
        text: str,
        store_turn: bool = True,
        think: bool | None = None,
        verbose: bool | None = None,
        json_mode: bool | None = None,
        attachments: list[dict] | None = None,
        on_chunk=None,
        on_thinking_chunk=None,
        on_debug_event=None,
        cancel_requested=None,
    ) -> ApiReply:
        payload = {"text": str(text or ""), "store_turn": bool(store_turn)}
        if attachments:
            payload["attachments"] = [dict(item) for item in list(attachments or []) if isinstance(item, dict)]
        if think is not None:
            payload["think"] = bool(think)
        if verbose is not None:
            payload["verbose"] = bool(verbose)
        if json_mode is not None:
            payload["json_mode"] = bool(json_mode)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "Cache-Control": "no-cache, no-store",
            "Pragma": "no-cache",
        }
        headers.update(self._account_headers())
        api_access_key = get_api_access_key()
        if api_access_key:
            headers["X-MMis-Access-Key"] = api_access_key
        token = get_auth_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib_request.Request(
            self._url("/chat/stream"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        stats: dict = {}
        model = self._runtime_model_cache
        parameters: dict | None = None
        summary: str | None = None
        debug_trace: dict | None = None
        memory_debug_snapshot: dict | None = None
        chunk_count = 0
        thinking_chunk_count = 0
        LOGGER.debug(
            "ui_stream_start path=/chat/stream store_turn=%s think=%s verbose=%s json_mode=%s text_chars=%d attachments=%d",
            bool(store_turn),
            think if think is not None else "runtime",
            verbose if verbose is not None else "runtime",
            json_mode if json_mode is not None else "runtime",
            len(str(text or "")),
            len(payload.get("attachments") or []),
        )

        try:
            with urllib_request.urlopen(req, timeout=self.stream_timeout_sec) as resp:
                for raw in resp:
                    if cancel_requested and cancel_requested():
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    event = self._parse_json(line)
                    kind = str(event.get("event") or "")
                    data = event.get("data")

                    if kind == "chunk":
                        piece = str(data or "")
                        answer_parts.append(piece)
                        chunk_count += 1
                        if on_chunk:
                            on_chunk(piece)
                    elif kind == "thinking":
                        piece = str(data or "")
                        thinking_parts.append(piece)
                        thinking_chunk_count += 1
                        if on_thinking_chunk:
                            on_thinking_chunk(piece)
                    elif kind == "memory_debug":
                        # Live memory-debug event
                        if on_debug_event:
                            try:
                                payload_debug = self._parse_json(str(data or "{}"))
                                on_debug_event(payload_debug)
                            except Exception:
                                pass
                    elif kind == "error":
                        LOGGER.warning("ui stream error event=%s", data)
                        raise ApiClientError(str(data or "Unknown API stream error"))
                    elif kind == "final" and isinstance(data, dict):
                        answer = str(data.get("answer") or "")
                        thinking = str(data.get("thinking") or "")
                        stats = data.get("stats") or {}
                        model = str(data.get("model") or model)
                        parameters = data.get("parameters") if isinstance(data.get("parameters"), dict) else None
                        raw_summary = data.get("summary")
                        summary = str(raw_summary).strip() if raw_summary is not None else None
                        if summary == "":
                            summary = None
                        debug_trace = data.get("debug_trace") if isinstance(data.get("debug_trace"), dict) else None
                        memory_debug_snapshot = (
                            data.get("memory_debug_snapshot")
                            if isinstance(data.get("memory_debug_snapshot"), dict)
                            else None
                        )
                        self._runtime_model_cache = model or self._runtime_model_cache
                        LOGGER.debug(
                            "ui_stream_done model=%s chunks=%d thinking_chunks=%d answer_chars=%d thinking_chars=%d",
                            model,
                            chunk_count,
                            thinking_chunk_count,
                            len(answer),
                            len(thinking),
                        )
                        return ApiReply(
                            answer=answer,
                            thinking=thinking,
                            thinking_generated=thinking_chunk_count > 0,
                            stats=stats,
                            model=model,
                            parameters=parameters,
                            summary=summary,
                            debug_trace=debug_trace,
                            memory_debug_snapshot=memory_debug_snapshot,
                        )
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            detail = raw
            try:
                payload_err = self._parse_json(raw)
                detail = str(payload_err.get("detail") or raw or exc)
            except Exception:
                detail = raw or str(exc)
            LOGGER.warning("ui stream http error code=%s detail=%s", exc.code, detail)
            raise ApiClientError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            LOGGER.warning("ui stream connection error reason=%s", exc.reason)
            raise ApiClientError(f"API connection failed: {exc.reason}") from exc
        except ApiClientError:
            raise
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("ui stream unexpected error")
            raise ApiClientError(str(exc)) from exc

        LOGGER.debug(
            "ui_stream_incomplete model=%s chunks=%d thinking_chunks=%d answer_chars=%d thinking_chars=%d",
            model,
            chunk_count,
            thinking_chunk_count,
            len("".join(answer_parts)),
            len("".join(thinking_parts)),
        )
        return ApiReply(
            answer="".join(answer_parts),
            thinking="".join(thinking_parts),
            thinking_generated=thinking_chunk_count > 0,
            stats=stats,
            model=model,
            parameters=parameters,
            summary=summary,
            debug_trace=debug_trace,
            memory_debug_snapshot=memory_debug_snapshot,
        )
