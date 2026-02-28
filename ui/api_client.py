"""HTTP client used by desktop UI to work with MMis API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import request as urllib_request


class ApiClientError(RuntimeError):
    pass


@dataclass
class ApiReply:
    answer: str
    thinking: str
    stats: dict
    model: str


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout_sec: float = 2.5, stream_timeout_sec: float = 600.0):
        env_url = os.getenv("MMIS_API_URL", "http://127.0.0.1:8000")
        self.base_url = (base_url or env_url).rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.stream_timeout_sec = float(stream_timeout_sec)
        self._runtime_model_cache = str(os.getenv("MMIS_MODEL_NAME", "")).strip()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not text:
            return {}
        return json.loads(text)

    def _request_json(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib_request.Request(self._url(path), data=data, headers=headers, method=method.upper())
        try:
            with urllib_request.urlopen(req, timeout=timeout or self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return self._parse_json(raw)
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            detail = raw
            try:
                payload = self._parse_json(raw)
                detail = str(payload.get("detail") or raw or exc)
            except Exception:
                detail = raw or str(exc)
            raise ApiClientError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise ApiClientError(f"API connection failed: {exc.reason}") from exc
        except Exception as exc:  # pragma: no cover
            raise ApiClientError(str(exc)) from exc

    def health(self) -> dict:
        payload = self._request_json("GET", "/health")
        self._runtime_model_cache = str(payload.get("model") or self._runtime_model_cache)
        return payload

    def list_models(self) -> dict:
        payload = self._request_json("GET", "/models")
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
        on_chunk=None,
        on_thinking_chunk=None,
        cancel_requested=None,
    ) -> ApiReply:
        payload = {"text": str(text or ""), "store_turn": bool(store_turn)}
        if think is not None:
            payload["think"] = bool(think)

        req = urllib_request.Request(
            self._url("/chat/stream"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST",
        )

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        stats: dict = {}
        model = self._runtime_model_cache

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
                        if on_chunk:
                            on_chunk(piece)
                    elif kind == "thinking":
                        piece = str(data or "")
                        thinking_parts.append(piece)
                        if on_thinking_chunk:
                            on_thinking_chunk(piece)
                    elif kind == "error":
                        raise ApiClientError(str(data or "Unknown API stream error"))
                    elif kind == "final" and isinstance(data, dict):
                        answer = str(data.get("answer") or "")
                        thinking = str(data.get("thinking") or "")
                        stats = data.get("stats") or {}
                        model = str(data.get("model") or model)
                        self._runtime_model_cache = model or self._runtime_model_cache
                        return ApiReply(answer=answer, thinking=thinking, stats=stats, model=model)
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            detail = raw
            try:
                payload = self._parse_json(raw)
                detail = str(payload.get("detail") or raw or exc)
            except Exception:
                detail = raw or str(exc)
            raise ApiClientError(f"HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise ApiClientError(f"API connection failed: {exc.reason}") from exc
        except ApiClientError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ApiClientError(str(exc)) from exc

        return ApiReply(
            answer="".join(answer_parts),
            thinking="".join(thinking_parts),
            stats=stats,
            model=model,
        )
