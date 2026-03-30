from __future__ import annotations

from llm.tokenizer import estimate_tokens
from ui.chat_window import ChatWindow


def test_complete_verbose_stats_fills_missing_verbose_fields() -> None:
    user_text = "как твои дела?"
    answer_text = "вроде нормально, а у тебя?"
    stats = {
        "answer_ms": 2142.0,
        "total_duration_ms": 2142.0,
        "prompt_eval_duration_ms": 262.0,
        "verbose_enabled": True,
    }

    out = ChatWindow._complete_verbose_stats(
        stats,
        user_text=user_text,
        answer_text=answer_text,
        fallback_elapsed_ms=2142,
        verbose_enabled=True,
    )

    assert int(out.get("prompt_eval_count") or 0) == estimate_tokens(user_text)
    assert int(out.get("eval_count") or 0) == estimate_tokens(answer_text)
    assert int(float(out.get("eval_duration_ms") or 0.0)) == 1880
    assert float(out.get("eval_tokens_per_sec") or 0.0) > 0.0


def test_complete_verbose_stats_uses_debug_trace_and_local_timers() -> None:
    out = ChatWindow._complete_verbose_stats(
        {"answer_ms": 25858.0, "verbose_enabled": True},
        user_text="как твои дела?",
        answer_text="вроде нормально, а у тебя?",
        fallback_elapsed_ms=25858,
        local_answer_ms=25111,
        local_thinking_ms=747,
        verbose_enabled=True,
        debug_trace={"prompt_pack": {"token_estimate": 512}},
    )

    assert int(out.get("prompt_eval_count") or 0) == 512
    assert int(out.get("eval_count") or 0) == estimate_tokens("вроде нормально, а у тебя?")
    assert int(float(out.get("prompt_eval_duration_ms") or 0.0)) == 747
    assert int(float(out.get("eval_duration_ms") or 0.0)) == 25111
    assert float(out.get("eval_tokens_per_sec") or 0.0) > 0.0


def test_perf_from_stats_returns_full_verbose_chip_row() -> None:
    stats = {
        "total_duration_ms": 2142.0,
        "eval_duration_ms": 1880.0,
        "eval_tokens_per_sec": 16.5,
        "prompt_eval_count": 512,
        "eval_count": 31,
        "verbose_enabled": True,
    }

    perf, stat_line = ChatWindow._perf_from_stats(stats)

    assert perf == ["2.1 s", "write 1.9 s", "16.5 tok/s", "prompt 512", "gen 31"]
    assert stat_line is not None
    assert "prompt 512" in stat_line


def test_perf_from_stats_forces_five_verbose_metrics() -> None:
    stats = ChatWindow._complete_verbose_stats(
        {"answer_ms": 25858.0, "verbose_enabled": True},
        user_text="как твои дела?",
        answer_text="вроде нормально, а у тебя?",
        fallback_elapsed_ms=25858,
        local_answer_ms=25111,
        local_thinking_ms=747,
        verbose_enabled=True,
        debug_trace={"prompt_pack": {"token_estimate": 512}},
    )

    perf, _stat_line = ChatWindow._perf_from_stats(stats, fallback_elapsed_ms=25858)

    assert perf == [
        "25.9 s",
        "write 25.1 s",
        f"{float(stats['eval_tokens_per_sec']):.1f} tok/s",
        "prompt 512",
        f"gen {estimate_tokens('вроде нормально, а у тебя?')}",
    ]


def test_stat_line_preserves_separate_thinking_ms() -> None:
    stat_line = ChatWindow._compose_stat_line(
        ["2.1 s", "write 1.9 s", "16.5 tok/s", "prompt 512", "gen 31"],
        "0.3 s",
    )

    assert ChatWindow._extract_thinking_ms_from_stat_line(stat_line) == "0.3 s"
    assert ChatWindow._split_stat_line(stat_line) == [
        "2.1 s",
        "write 1.9 s",
        "16.5 tok/s",
        "prompt 512",
        "gen 31",
    ]


def test_stat_line_without_prefix_does_not_infer_thinking_ms_from_total_time() -> None:
    stat_line = "25.9 s | write 25.1 s | 6.3 tok/s | prompt 2608 | gen 73"

    assert ChatWindow._extract_thinking_ms_from_stat_line(stat_line) == ""


def test_resolve_thinking_text_ignores_final_thinking_without_stream_generation() -> None:
    assert ChatWindow._resolve_thinking_text("hidden reasoning", "", False) == ""
    assert ChatWindow._resolve_thinking_text("hidden reasoning", "streamed reasoning", False) == "streamed reasoning"
    assert ChatWindow._resolve_thinking_text("hidden reasoning", "", True) == "hidden reasoning"


def test_upgrade_legacy_verbose_stat_line_without_thinking_keeps_timer_hidden() -> None:
    window = ChatWindow.__new__(ChatWindow)
    window._verbose_enabled = True

    stat_line = window._upgrade_legacy_verbose_stat_line(
        role="ai",
        text="вроде нормально, а у тебя?",
        stat_line="25858 ms",
        thinking=None,
        user_text="как твои дела?",
    )

    assert window._extract_thinking_ms_from_stat_line(stat_line) == ""


def test_upgrade_legacy_verbose_stat_line_backfills_full_row() -> None:
    window = ChatWindow.__new__(ChatWindow)
    window._verbose_enabled = True

    stat_line = window._upgrade_legacy_verbose_stat_line(
        role="ai",
        text="вроде нормально, а у тебя?",
        stat_line="25858 ms",
        thinking="думание",
        user_text="как твои дела?",
    )

    assert window._extract_thinking_ms_from_stat_line(stat_line) == ""
    assert window._split_stat_line(stat_line) == [
        "25.9 s",
        "write 25.9 s",
        f"{float(estimate_tokens('вроде нормально, а у тебя?') / 25.858):.1f} tok/s",
        f"prompt {estimate_tokens('как твои дела?')}",
        f"gen {estimate_tokens('вроде нормально, а у тебя?')}",
    ]


def test_should_follow_stream_scroll_only_when_chat_already_overflowed_and_is_near_bottom() -> None:
    assert not ChatWindow._should_follow_stream_scroll(0, 0)
    assert not ChatWindow._should_follow_stream_scroll(12, 120)
    assert ChatWindow._should_follow_stream_scroll(108, 120)
    assert ChatWindow._should_follow_stream_scroll(96, 120, margin_px=24)


def test_display_stats_prefer_visible_answer_metrics_over_reasoning_included_backend_values() -> None:
    answer_text = "Приятно, что всё хорошо. Суши вдруг вспомнил?"
    stats = ChatWindow._complete_verbose_stats(
        {
            "total_duration_ms": 106200.0,
            "eval_duration_ms": 88500.0,
            "eval_tokens_per_sec": 5.7,
            "prompt_eval_count": 2754,
            "eval_count": 501,
            "verbose_enabled": True,
        },
        user_text="тоже хорошо, спасибо",
        answer_text=answer_text,
        fallback_elapsed_ms=106200,
        local_answer_ms=4300,
        local_thinking_ms=51900,
        verbose_enabled=True,
    )

    perf, _ = ChatWindow._perf_from_stats(stats, fallback_elapsed_ms=106200)

    assert ChatWindow._thinking_ms_label(stats) == "51.9 s"
    assert perf == [
        "106.2 s",
        "write 4.3 s",
        f"{float(stats['display_tok_s']):.1f} tok/s",
        "prompt 2754",
        f"gen {estimate_tokens(answer_text)}",
    ]


def test_display_total_prefers_real_wait_when_backend_total_is_smaller_than_thinking() -> None:
    answer_text = "В парке лаял пёс, а рядом крадётся белка."
    stats = ChatWindow._complete_verbose_stats(
        {
            "total_duration_ms": 48900.0,
            "eval_duration_ms": 14700.0,
            "eval_tokens_per_sec": 12.4,
            "prompt_eval_count": 2661,
            "eval_count": 183,
            "verbose_enabled": True,
        },
        user_text="прикольно, а ещё",
        answer_text=answer_text,
        fallback_elapsed_ms=85100,
        local_answer_ms=14700,
        local_thinking_ms=70400,
        verbose_enabled=True,
    )

    perf, _ = ChatWindow._perf_from_stats(stats, fallback_elapsed_ms=85100)

    assert ChatWindow._thinking_ms_label(stats) == "70.4 s"
    assert int(float(stats.get("display_elapsed_ms") or 0) or 0) == 85100
    assert perf == [
        "85.1 s",
        "write 14.7 s",
        f"{float(stats['display_tok_s']):.1f} tok/s",
        "prompt 2661",
        f"gen {estimate_tokens(answer_text)}",
    ]


def test_display_total_uses_wall_clock_instead_of_summing_thinking_and_write() -> None:
    stats = ChatWindow._complete_verbose_stats(
        {
            "total_duration_ms": 18000.0,
            "eval_duration_ms": 5000.0,
            "verbose_enabled": True,
        },
        user_text="tool test",
        answer_text="готово",
        fallback_elapsed_ms=40000,
        local_answer_ms=5000,
        local_thinking_ms=7000,
        verbose_enabled=True,
    )

    assert int(float(stats.get("display_elapsed_ms") or 0) or 0) == 40000
