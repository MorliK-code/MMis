from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.fact_extractor import FactExtractor
from memory.ingest_analyzer import IngestAnalysis, analyze_message_for_memory
from memory.memory_models import FactRecordV2, MemoryScope


def _configure_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except Exception:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug one message through memory ingest analysis and fact extraction."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Message text to inspect. If omitted, the script reads UTF-8 text from stdin.",
    )
    parser.add_argument(
        "--speaker",
        default="user",
        help="Speaker role for fact extraction: user, assistant, system. Default: user",
    )
    parser.add_argument(
        "--scope",
        default="conversation",
        help="Memory scope: conversation, project, session, temporary, global_user, character. Default: conversation",
    )
    parser.add_argument(
        "--mode",
        default="BALANCED",
        help="Fact extraction mode: FAST, BALANCED, QUALITY. Default: BALANCED",
    )
    parser.add_argument(
        "--namespace",
        default="debug",
        help="Namespace to stamp into fact extraction metadata. Default: debug",
    )
    parser.add_argument(
        "--event-id",
        default="debug:message",
        help="Source event id to stamp into final facts. Default: debug:message",
    )
    parser.add_argument("--json", action="store_true", help="Print pure JSON instead of human-readable blocks")
    parser.add_argument(
        "--metadata",
        default="",
        help="Optional metadata JSON object to pass into ingest analysis and fact extraction",
    )
    return parser


def _parse_scope(value: str) -> MemoryScope:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    for item in MemoryScope:
        if item.value == token:
            return item
    raise ValueError(f"Unsupported scope: {value}")


def _parse_metadata(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("--metadata must be a JSON object")
    return dict(value)


def _read_text(args: argparse.Namespace) -> str:
    if list(args.text or []):
        return " ".join(str(x) for x in list(args.text or [])).strip()
    if not sys.stdin.isatty():
        return str(sys.stdin.read() or "").strip()
    return ""


def _analysis_payload(analysis: IngestAnalysis) -> dict[str, Any]:
    views = dict(analysis.memory_views or {})
    return {
        "raw": str(analysis.raw_text or ""),
        "normalized": str(analysis.normalized_text or ""),
        "canonical": str(analysis.canonical_text or ""),
        "search_text": str(analysis.search_text or ""),
        "entity_keys": list(views.get("entity_keys") or []),
        "numeric_keys": list(views.get("numeric_keys") or []),
        "entities": [item.to_dict() for item in list(analysis.entities or [])],
        "numeric_facts": [item.to_dict() for item in list(analysis.numeric_facts or [])],
        "stable_facts": [item.to_dict() for item in list(analysis.stable_facts or [])],
        "emotion": (analysis.emotion.to_dict() if analysis.emotion is not None else None),
        "memory_tags": [str(tag) for tag in list(analysis.tags or [])],
        "memory_views": views,
    }


def _facts_payload(rows: list[FactRecordV2]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in list(rows or [])]


def _debug_payload(
    *,
    text: str,
    speaker: str,
    scope: MemoryScope,
    mode: str,
    namespace: str,
    event_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    analysis_metadata = dict(metadata or {})
    analysis = analyze_message_for_memory(text, metadata=analysis_metadata)
    extractor = FactExtractor()
    facts = extractor.extract_v2(
        text=text,
        metadata={
            **dict(metadata or {}),
            "event_id": str(event_id or "debug:message"),
            "namespace": str(namespace or "debug"),
        },
        speaker=str(speaker or "user"),
        scope=scope,
        mode=str(mode or "BALANCED"),
        analysis=analysis,
    )
    return {
        "input": {
            "text": str(text or ""),
            "speaker": str(speaker or "user"),
            "scope": str(scope.value),
            "mode": str(mode or "BALANCED"),
            "namespace": str(namespace or "debug"),
            "event_id": str(event_id or "debug:message"),
            "metadata": dict(metadata or {}),
        },
        "analysis": _analysis_payload(analysis),
        "final_facts": _facts_payload(facts),
    }


def _print_human(payload: dict[str, Any]) -> None:
    analysis = dict(payload.get("analysis") or {})
    final_facts = list(payload.get("final_facts") or [])
    print("=" * 100)
    print("Input")
    print(json.dumps(dict(payload.get("input") or {}), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Raw")
    print(str(analysis.get("raw") or ""))
    print("=" * 100)
    print("Normalized")
    print(str(analysis.get("normalized") or ""))
    print("=" * 100)
    print("Canonical")
    print(str(analysis.get("canonical") or ""))
    print("=" * 100)
    print("Search Text")
    print(str(analysis.get("search_text") or ""))
    print("=" * 100)
    print("Entity Keys")
    print(json.dumps(list(analysis.get("entity_keys") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Numeric Keys")
    print(json.dumps(list(analysis.get("numeric_keys") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Entities")
    print(json.dumps(list(analysis.get("entities") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Numeric Facts")
    print(json.dumps(list(analysis.get("numeric_facts") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Stable Facts")
    print(json.dumps(list(analysis.get("stable_facts") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Memory Tags")
    print(json.dumps(list(analysis.get("memory_tags") or []), ensure_ascii=False, indent=2))
    print("=" * 100)
    print("Final Facts")
    print(json.dumps(final_facts, ensure_ascii=False, indent=2))


def main() -> int:
    _configure_stdout()
    args = _build_parser().parse_args()
    text = _read_text(args)
    if not text:
        print("No message text provided. Pass text as arguments or via stdin.", file=sys.stderr)
        return 2
    try:
        scope = _parse_scope(str(args.scope or "conversation"))
        metadata = _parse_metadata(str(args.metadata or ""))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    payload = _debug_payload(
        text=text,
        speaker=str(args.speaker or "user"),
        scope=scope,
        mode=str(args.mode or "BALANCED"),
        namespace=str(args.namespace or "debug"),
        event_id=str(args.event_id or "debug:message"),
        metadata=metadata,
    )
    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
