from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import tempfile
import unittest
from pathlib import Path

from scripts.memory_backfill_answer_only import main as backfill_main


_FORMATTED_ASSISTANT = (
    "[PARAMETERS]\n"
    "debug=I was born in 1999\n\n"
    "[SUMMARY]\n"
    "short\n\n"
    "[RESPONSE]\n"
    "I can help with setup."
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_memory_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    events = [
        {
            "event_id": "e1",
            "ts": "2026-03-05T10:00:00+02:00",
            "type": "user_message",
            "payload": {
                "role": "user",
                "text": "I was born in 2003",
                "metadata": {"lang": "en"},
                "ids": {},
            },
            "tags": [],
            "trace_id": "",
            "model": "",
            "latency_ms": 0.0,
        },
        {
            "event_id": "e2",
            "ts": "2026-03-05T10:00:02+02:00",
            "type": "user_message",
            "payload": {
                "role": "user",
                "text": "I was born in 2003",
                "metadata": {"lang": "en"},
                "ids": {},
            },
            "tags": [],
            "trace_id": "",
            "model": "",
            "latency_ms": 0.0,
        },
        {
            "event_id": "e3",
            "ts": "2026-03-05T10:00:05+02:00",
            "type": "assistant_message",
            "payload": {
                "role": "assistant",
                "text": _FORMATTED_ASSISTANT,
                "metadata": {"lang": "en"},
                "ids": {},
            },
            "tags": [],
            "trace_id": "",
            "model": "",
            "latency_ms": 0.0,
        },
    ]
    events_path = root / "events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in events) + "\n",
        encoding="utf-8",
    )

    _write_json(
        root / "short_memory.json",
        {
            "items": [
                {"id": "e2", "role": "user", "type": "message", "text": "I was born in 2003", "meta": {}},
                {"id": "e3", "role": "assistant", "type": "message", "text": _FORMATTED_ASSISTANT, "meta": {}},
            ],
            "rolling_summary": "",
            "rolling_summary_meta": {},
        },
    )
    _write_json(
        root / "long_memory_docs.json",
        {
            "docs": [
                {
                    "id": "doc-chat-1",
                    "text": _FORMATTED_ASSISTANT,
                    "thinking": "",
                    "created_at": 1.0,
                    "updated_at": "2026-03-05T10:00:06+02:00",
                    "source": "chat",
                    "tags": [],
                    "importance": 0.6,
                    "confidence": 0.7,
                    "meta": {"role": "assistant"},
                },
                {
                    "id": "doc-fact-old",
                    "text": "assistant.birth_year=1999",
                    "thinking": "",
                    "created_at": 1.0,
                    "updated_at": "2026-03-05T10:00:07+02:00",
                    "source": "fact",
                    "tags": ["fact", "subject_assistant", "key_birth_year"],
                    "importance": 0.8,
                    "confidence": 0.9,
                    "meta": {"fact": {"subject": "assistant", "key": "birth_year", "value": "1999"}},
                },
            ]
        },
    )
    old_embedding = [0.5] + [0.0] * 31
    _write_json(
        root / "vector_store.json",
        {
            "records": [
                {
                    "id": "msg:e3",
                    "text": _FORMATTED_ASSISTANT,
                    "embedding": list(old_embedding),
                    "metadata": {"type": "message", "role": "assistant", "event_id": "e3", "source": "text"},
                },
                {
                    "id": "doc:doc-chat-1",
                    "text": _FORMATTED_ASSISTANT,
                    "embedding": list(old_embedding),
                    "metadata": {"type": "summary", "role": "assistant", "doc_id": "doc-chat-1", "source": "chat"},
                },
                {
                    "id": "fact:doc-fact-old",
                    "text": "assistant.birth_year=1999",
                    "embedding": list(old_embedding),
                    "metadata": {"type": "fact", "source": "fact", "doc_id": "doc-fact-old"},
                },
            ]
        },
    )
    _write_json(
        root / "user_profile_store.json",
        {"profiles": {}, "versions": {}, "pending_facts": {}, "confirmed_facts": {}},
    )
    _write_json(
        root / "assistant_profile_store.json",
        {
            "profiles": {
                "default": {
                    "birth_year": {
                        "value": "1999",
                        "confidence": 0.9,
                        "source_event_id": "e3",
                        "source": "fact_add",
                        "updated_at": "2026-03-05T10:00:08+02:00",
                        "needs_confirmation": False,
                        "pending": [],
                    }
                }
            },
            "versions": {"default": {"birth_year": []}},
            "pending_facts": {"default": {}},
            "confirmed_facts": {
                "default": {
                    "birth_year": {
                        "key": "birth_year",
                        "value": "1999",
                        "op": "add",
                        "confidence": 0.9,
                        "count": 2,
                        "evidence": [],
                        "source_event_ids": ["e3"],
                        "confirmed_at": "2026-03-05T10:00:08+02:00",
                        "status": "confirmed",
                    }
                }
            },
        },
    )


class MemoryBackfillAnswerOnlyTests(unittest.TestCase):
    def test_cleanup_creates_backup_and_reembeds_vectors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_backfill_answer_only_") as tmpdir:
            root = Path(tmpdir)
            _seed_memory_dir(root)
            backup_root = root / "backup"
            rc = backfill_main(
                [
                    "--memory-dir",
                    str(root),
                    "--backup-dir",
                    str(backup_root),
                    "--yes",
                ]
            )
            self.assertEqual(rc, 0)

            backups = list(backup_root.glob("answer_only_*"))
            self.assertTrue(backups)
            self.assertTrue((backups[0] / "events.jsonl").exists())
            self.assertTrue((backups[0] / "vector_store.json").exists())

            events_lines = (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            events = [json.loads(x) for x in events_lines if x.strip()]
            assistant_event = next(x for x in events if str(x.get("event_id")) == "e3")
            self.assertEqual(
                str(dict(assistant_event.get("payload") or {}).get("text") or ""),
                "I can help with setup.",
            )

            short_payload = json.loads((root / "short_memory.json").read_text(encoding="utf-8"))
            assistant_short = next(x for x in list(short_payload.get("items") or []) if str(x.get("id")) == "e3")
            self.assertEqual(str(assistant_short.get("text") or ""), "I can help with setup.")

            long_payload = json.loads((root / "long_memory_docs.json").read_text(encoding="utf-8"))
            chat_doc = next(x for x in list(long_payload.get("docs") or []) if str(x.get("id")) == "doc-chat-1")
            self.assertEqual(str(chat_doc.get("text") or ""), "I can help with setup.")

            vector_payload = json.loads((root / "vector_store.json").read_text(encoding="utf-8"))
            msg_record = next(x for x in list(vector_payload.get("records") or []) if str(x.get("id")) == "msg:e3")
            self.assertEqual(str(msg_record.get("text") or ""), "I can help with setup.")
            self.assertNotEqual(list(msg_record.get("embedding") or []), [0.5] + [0.0] * 31)

    def test_dry_run_does_not_modify_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_backfill_answer_only_dry_") as tmpdir:
            root = Path(tmpdir)
            _seed_memory_dir(root)
            before = {name: (root / name).read_text(encoding="utf-8") for name in (
                "events.jsonl",
                "short_memory.json",
                "long_memory_docs.json",
                "vector_store.json",
                "user_profile_store.json",
                "assistant_profile_store.json",
            )}
            rc = backfill_main(
                [
                    "--memory-dir",
                    str(root),
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            after = {name: (root / name).read_text(encoding="utf-8") for name in before.keys()}
            self.assertEqual(before, after)

    def test_rebuild_facts_removes_parameter_artifacts_and_keeps_valid_fact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_backfill_answer_only_rebuild_") as tmpdir:
            root = Path(tmpdir)
            _seed_memory_dir(root)
            rc = backfill_main(
                [
                    "--memory-dir",
                    str(root),
                    "--yes",
                    "--rebuild-facts",
                ]
            )
            self.assertEqual(rc, 0)

            assistant_profile = json.loads((root / "assistant_profile_store.json").read_text(encoding="utf-8"))
            assistant_default = dict(dict(assistant_profile.get("profiles") or {}).get("default") or {})
            self.assertNotIn("birth_year", assistant_default)

            user_profile = json.loads((root / "user_profile_store.json").read_text(encoding="utf-8"))
            user_default = dict(dict(user_profile.get("profiles") or {}).get("default") or {})
            self.assertEqual(str(dict(user_default.get("birth_year") or {}).get("value") or ""), "2003")

            long_payload = json.loads((root / "long_memory_docs.json").read_text(encoding="utf-8"))
            long_texts = [str(x.get("text") or "") for x in list(long_payload.get("docs") or [])]
            self.assertFalse(any("assistant.birth_year=1999" in text for text in long_texts))
            self.assertTrue(any("user.birth_year=2003" in text for text in long_texts))

            vector_payload = json.loads((root / "vector_store.json").read_text(encoding="utf-8"))
            vector_texts = [str(x.get("text") or "") for x in list(vector_payload.get("records") or [])]
            self.assertFalse(any("assistant.birth_year=1999" in text for text in vector_texts))
            self.assertTrue(any("user.birth_year=2003" in text for text in vector_texts))


if __name__ == "__main__":
    unittest.main()
