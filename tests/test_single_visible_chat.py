from __future__ import annotations

from ui.chat_sessions import SINGLE_VISIBLE_CHAT_ID, SINGLE_VISIBLE_CHAT_TITLE, collapse_to_single_visible_chat


def test_collapse_to_single_visible_chat_prefers_active_chat_history() -> None:
    chats = [
        {
            "id": "chat-1",
            "title": "one",
            "created_at": "2026-03-01T10:00:00",
            "updated_at": "2026-03-01T10:05:00",
            "history": [("user", "one", None, None, None)],
        },
        {
            "id": "chat-2",
            "title": "two",
            "created_at": "2026-03-02T10:00:00",
            "updated_at": "2026-03-02T10:05:00",
            "history": [("user", "two", None, None, None)],
        },
    ]

    out = collapse_to_single_visible_chat(chats, active_chat_id="chat-1")

    assert out["id"] == SINGLE_VISIBLE_CHAT_ID
    assert out["title"] == SINGLE_VISIBLE_CHAT_TITLE
    assert list(out["history"]) == [("user", "one", None, None, None)]


def test_collapse_to_single_visible_chat_falls_back_to_latest_chat() -> None:
    chats = [
        {
            "id": "chat-1",
            "title": "older",
            "created_at": "2026-03-01T10:00:00",
            "updated_at": "2026-03-01T10:05:00",
            "history": [("user", "older", None, None, None)],
        },
        {
            "id": "chat-2",
            "title": "newer",
            "created_at": "2026-03-02T10:00:00",
            "updated_at": "2026-03-02T10:05:00",
            "history": [("user", "newer", None, None, None)],
        },
    ]

    out = collapse_to_single_visible_chat(chats, active_chat_id=None)

    assert out["id"] == SINGLE_VISIBLE_CHAT_ID
    assert list(out["history"]) == [("user", "newer", None, None, None)]
