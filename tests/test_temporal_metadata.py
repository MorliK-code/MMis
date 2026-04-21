from __future__ import annotations

import datetime as dt

from core.character_runtime import CharacterRuntime
from core.response_pipeline import PipelineContext, _inject_temporal_grounding, _request_metadata, _zoneinfo_or_utc
from prompt_engine.prompt_engine import PromptEngine
from prompt_engine.token_budget_manager import TokenBudgetManager


def test_metadata_block_keeps_temporal_grounding_under_default_budget() -> None:
    context_tags = {
        "lang": "ru",
        "intent": "chat",
        "mood": "neutral",
        "active_mode": "chatting",
        "local_date": "2026-04-21",
        "local_region": "RTZ 2",
        "greeting_allowed": "false",
        "dialog_sarcasm_level": "0.290",
        "dialog_warmth_level": "0.560",
        "dialog_strictness_level": "0.710",
        "dialog_verbosity_level": "0.380",
        "is_technical": "false",
        "use_term_now": "false",
        "now_human": "21.04.2026 12:42:16",
        "time_human": "12:42:16",
        "today_human": "21.04.2026",
        "timezone": "Europe/Kiev",
        "now_iso": "2026-04-21T12:42:16+03:00",
    }
    legacy_tags_block = "\n".join(
        [
            "- lang: ru",
            "- mood: neutral",
            "- intent: chat",
            "- active_mode: chatting",
            "- local_date: 2026-04-21",
            "- is_technical: false",
            "- use_term_now: false",
        ]
    )

    block = PromptEngine._build_metadata_block(
        state_map={"context_tags": context_tags},
        blocks={"context_tags": legacy_tags_block},
    )
    clipped = TokenBudgetManager().truncate(block, 80)

    assert "- now_human: 21.04.2026 12:42:16" in clipped
    assert "- time_human: 12:42:16" in clipped
    assert "- today_human: 21.04.2026" in clipped
    assert "- timezone: Europe/Kiev" in clipped
    assert clipped.index("- now_human:") < clipped.index("- local_date:")


def test_temporal_grounding_reaches_request_metadata_and_state_tags() -> None:
    ctx = PipelineContext(
        route="chat",
        user_msg="what time is it?",
        state={"context_tags": {}, "cooldowns": {}},
        meta={"timezone": "Europe/Kiev"},
        retrieved_memories=[],
        traits={},
        policies={},
        profile="BALANCED",
    )

    _inject_temporal_grounding(ctx)
    req_metadata = _request_metadata(ctx)
    state_tags = dict(ctx.state.get("context_tags") or {})

    for key in ("now_iso", "now_human", "today_human", "time_human", "timezone"):
        assert str(ctx.meta.get(key) or "").strip()
        assert str(state_tags.get(key) or "").strip()
        assert str(req_metadata.get(key) or "").strip()


def test_zoneinfo_fallback_uses_system_local_timezone() -> None:
    tzinfo = _zoneinfo_or_utc("Invalid/DefinitelyMissing")
    now = dt.datetime.now()
    local_offset = dt.datetime.now().astimezone().utcoffset()

    assert tzinfo.utcoffset(now) == local_offset


def test_zoneinfo_accepts_fixed_utc_offset() -> None:
    tzinfo = _zoneinfo_or_utc("UTC+03:00")

    assert tzinfo.utcoffset(dt.datetime.now()) == dt.timedelta(hours=3)


def test_character_runtime_keeps_temporal_tags_in_context_block() -> None:
    pack = CharacterRuntime.build(
        state={
            "context_tags": {
                "local_date": "2026-04-21",
                "now_human": "21.04.2026 12:42:16",
                "time_human": "12:42:16",
                "today_human": "21.04.2026",
                "timezone": "Europe/Kiev",
                "now_iso": "2026-04-21T12:42:16+03:00",
            }
        },
        user_msg="time?",
        retrieved_memories=[],
        traits={},
        policies={},
    )

    block = str(pack.blocks.get("context_tags") or "")

    assert "- now_human: 21.04.2026 12:42:16" in block
    assert "- time_human: 12:42:16" in block
    assert "- timezone: Europe/Kiev" in block
