from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_SECTION_HEADER_RE = re.compile(r"(?im)^\s*\[(PARAMETERS|SUMMARY|RESPONSE)\]\s*$")


@dataclass(frozen=True)
class AssistantTextSanitizeResult:
    text: str
    reason: str
    changed: bool
    had_service_sections: bool


def clean_assistant_text_for_memory(result: Any) -> AssistantTextSanitizeResult:
    structured_output = _coerce_dict(getattr(result, "structured_output", {}))
    text = _coerce_text(getattr(result, "text", ""))
    return sanitize_assistant_memory_text(text=text, structured_output=structured_output)


def sanitize_assistant_memory_text(
    *,
    text: str,
    structured_output: dict[str, Any] | None = None,
) -> AssistantTextSanitizeResult:
    source = _coerce_text(text)
    normalized_source = _normalize_memory_text(source)
    had_sections = contains_memory_service_sections(source)

    structured = _coerce_text(_coerce_dict(structured_output).get("text"))
    if structured:
        normalized = _normalize_memory_text(structured)
        return AssistantTextSanitizeResult(
            text=normalized,
            reason="structured_output_text",
            changed=(normalized != normalized_source),
            had_service_sections=had_sections,
        )

    response = _extract_response_block(source)
    if response:
        normalized = _normalize_memory_text(response)
        return AssistantTextSanitizeResult(
            text=normalized,
            reason="response_block_extract",
            changed=(normalized != normalized_source),
            had_service_sections=had_sections,
        )

    fallback = _normalize_memory_text(_strip_parameters_and_summary(source))
    if had_sections:
        reason = "fallback_strip"
    else:
        reason = "plain_text"
    return AssistantTextSanitizeResult(
        text=fallback,
        reason=reason,
        changed=(fallback != normalized_source),
        had_service_sections=had_sections,
    )


def contains_memory_service_sections(text: str) -> bool:
    return bool(_SECTION_HEADER_RE.search(_coerce_text(text)))


def _extract_response_block(text: str) -> str:
    sections = _parse_known_sections(_coerce_text(text))
    return str(sections.get("RESPONSE") or "")


def _strip_parameters_and_summary(text: str) -> str:
    source = _coerce_text(text)
    matches = list(_SECTION_HEADER_RE.finditer(source))
    if not matches:
        return source

    chunks: list[str] = []
    cursor = 0
    size = len(matches)
    for idx, match in enumerate(matches):
        start = match.start()
        if start > cursor:
            chunks.append(source[cursor:start])
        end = matches[idx + 1].start() if (idx + 1) < size else len(source)
        section = str(match.group(1) or "").strip().upper()
        body = source[match.end() : end]
        if section == "RESPONSE":
            chunks.append(body)
        elif section in {"PARAMETERS", "SUMMARY"}:
            tail = _tail_after_service_block(body)
            if tail:
                chunks.append(tail)
        else:
            chunks.append(source[start:end])
        cursor = end
    if cursor < len(source):
        chunks.append(source[cursor:])
    return "".join(chunks)


def _parse_known_sections(text: str) -> dict[str, str]:
    source = _coerce_text(text)
    matches = list(_SECTION_HEADER_RE.finditer(source))
    if not matches:
        return {}

    out: dict[str, str] = {}
    size = len(matches)
    for idx, match in enumerate(matches):
        section = str(match.group(1) or "").strip().upper()
        end = matches[idx + 1].start() if (idx + 1) < size else len(source)
        body = _normalize_memory_text(source[match.end() : end])
        if not body:
            continue
        previous = str(out.get(section) or "")
        if previous:
            out[section] = f"{previous}\n\n{body}"
        else:
            out[section] = body
    return out


def _tail_after_service_block(block: str) -> str:
    text = _coerce_text(block).replace("\r\n", "\n").replace("\r", "\n")
    if not text or "\n\n" not in text:
        return ""
    parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(parts) < 2:
        return ""
    return "\n\n".join(parts[1:]).strip()


def _normalize_memory_text(value: Any) -> str:
    text = _coerce_text(value).replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return ""
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        return dict(value)
    except Exception:
        return {}


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()
