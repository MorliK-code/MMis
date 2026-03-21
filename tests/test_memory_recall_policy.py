from __future__ import annotations

from memory.recall_policy import classify_query_recall_profile


def test_classify_query_recall_profile_detects_russian_preference_claim_query() -> None:
    profile = classify_query_recall_profile("ну не скажи, я тебе говорил кто мне нравиться")

    assert profile.claim_like is True
    assert profile.spontaneous_claim is False
    assert profile.mode == ""


def test_classify_query_recall_profile_detects_russian_spontaneous_preference_query() -> None:
    profile = classify_query_recall_profile("что мне нравится")

    assert profile.claim_like is True
    assert profile.spontaneous_claim is True
    assert profile.mode == ""
