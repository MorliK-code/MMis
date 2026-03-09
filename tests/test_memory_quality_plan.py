from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from core.character_runtime import CharacterRuntime
from memory.auto_migration import MIGRATION_STATE_FILE, run_auto_migration
from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore, embed_text
from prompt_engine.token_budget_manager import ContextBlock, TokenBudget, TokenBudgetManager


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")


class MemoryQualityPlanTests(unittest.TestCase):
    def _new_manager(self, root: Path, *, facts_scope: str = "user_only") -> MemoryManager:
        return MemoryManager(
            short_memory=ShortMemory(path=root / "short_memory.json", autosave=False),
            long_memory=LongMemory(path=root / "long_memory_docs.json"),
            vector_store=VectorStore(path=root / "vector_store.json", dim=64),
            fact_extractor=FactExtractor(),
            user_profile_store=UserProfileStore(path=root / "user_profile_store.json"),
            assistant_profile_store=AssistantProfileStore(path=root / "assistant_profile_store.json"),
            event_store=EventStore(path=root / "events.jsonl"),
            facts_scope=facts_scope,
            include_pending_facts_in_retrieval=False,
            confirmation_ttl_sec=300,
        )

    def test_embedding_is_deterministic_across_processes(self) -> None:
        text = "Deterministic embedding text 42 hello"
        local = embed_text(text, dim=64)
        child_code = (
            "import json\n"
            "from memory.vector_store import embed_text\n"
            f"print(json.dumps(embed_text({text!r}, dim=64)))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", child_code],
            capture_output=True,
            text=True,
            check=True,
        )
        remote = json.loads(proc.stdout.strip())
        self.assertEqual(local, remote)

    def test_fact_extractor_name_pattern_is_strict(self) -> None:
        extractor = FactExtractor()
        noisy = extractor.extract(text="я здесь", metadata={}, speaker="user")
        clean = extractor.extract(text="меня зовут Алексей", metadata={}, speaker="user")
        noisy_names = [x for x in noisy if str(x.key) == "name"]
        clean_names = [x for x in clean if str(x.key) == "name"]
        self.assertFalse(noisy_names)
        self.assertTrue(clean_names)

    def test_user_only_scope_ignores_assistant_facts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_user_only_") as tmpdir:
            root = Path(tmpdir)
            manager = self._new_manager(root, facts_scope="user_only")
            manager.ingest_message("assistant", "my name is Luna", metadata={"lang": "en"})
            assistant_name = manager.assistant_profile_store.get("name", profile_id="default")
            fact_docs = [x for x in manager.long_memory.list_docs(limit=50) if str(x.source) == "fact"]
            self.assertIsNone(assistant_name)
            self.assertFalse(fact_docs)

    def test_pending_facts_not_returned_in_retrieval_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_pending_gate_") as tmpdir:
            root = Path(tmpdir)
            manager = self._new_manager(root)
            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            hits = manager.retrieve("birth year", k=8)
            fact_hits = [x for x in hits if str(x.source) == "fact"]
            self.assertFalse(fact_hits)

    def test_yes_with_context_confirms_only_expected_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_confirm_ctx_") as tmpdir:
            root = Path(tmpdir)
            manager = self._new_manager(root)
            manager.ingest_message("user", "I was born in 2003", metadata={"lang": "en"})
            manager.ingest_message("user", "I was born in 2004", metadata={"lang": "en"})
            manager.ingest_message("assistant", "You were born in 2003, right?", metadata={"lang": "en"})
            manager.ingest_message("user", "yes", metadata={"lang": "en"})
            current = manager.user_profile_store.get("birth_year", profile_id="default") or {}
            self.assertEqual(str(current.get("value") or ""), "2003")

    def test_prompt_memory_sort_uses_score_when_priority_missing(self) -> None:
        policies = {
            "budgets": {
                "total_tokens": 280,
                "system_tokens": 40,
                "persona_tokens": 24,
                "state_tokens": 24,
                "tags_tokens": 24,
                "memory_tokens": 90,
                "tail_tokens": 24,
                "user_tokens": 20,
                "output_schema_tokens": 16,
                "tail_turns": 2,
                "tail_turn_tokens": 12,
                "memory_item_tokens": 24,
            }
        }
        pack = CharacterRuntime.build(
            state={"history": []},
            user_msg="hi",
            retrieved_memories=[
                {"text": "low score memory", "score": 0.1, "source": "test"},
                {"text": "high score memory", "score": 0.9, "source": "test"},
            ],
            traits={},
            policies=policies,
        )
        lines = [x.strip() for x in str(pack.blocks.get("retrieved_memories") or "").splitlines() if x.strip().startswith("- ")]
        self.assertTrue(lines)
        self.assertIn("high score memory", lines[0])

    def test_token_pressure_keeps_memory_before_recent_chat(self) -> None:
        budget = TokenBudget(
            total=260,
            reserve=80,
            system_core=64,
            personality=20,
            rules=20,
            user_profile=10,
            metadata=8,
            tools_state=8,
            memory_retrieval=90,
            recent_chat=90,
            long_summary=10,
            output=10,
        )
        manager = TokenBudgetManager(budget=budget, chars_per_token=4.0)
        blocks = [
            ContextBlock(id="system_core", content="sys " * 200, bucket="system", priority=100, required=True),
            ContextBlock(id="memory_retrieval", content="memory " * 500, bucket="memory", priority=78, shrink_strategy="summarize", min_tokens=36),
            ContextBlock(id="recent_chat", content="chat " * 800, bucket="history", priority=66, shrink_strategy="summarize"),
            ContextBlock(id="user", content="what now " * 40, bucket="user", priority=100, required=True, min_tokens=12),
        ]
        fitted, _stats = manager.fit_context_blocks(blocks)
        self.assertTrue(str(fitted.get("memory_retrieval") or "").strip())

    def test_auto_migration_backup_version_and_idempotence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_auto_migrate_") as tmpdir:
            root = Path(tmpdir)
            self._seed_migration_fixture(root)
            first = run_auto_migration(
                memory_dir=root,
                target_schema_version=2,
                auto_on_start=True,
                facts_scope="user_only",
            )
            self.assertTrue(bool(first.get("ran")))
            self.assertTrue(bool(first.get("success")))
            backup_dir = Path(str(first.get("backup_dir") or ""))
            self.assertTrue(backup_dir.exists())
            self.assertTrue((backup_dir / "events.jsonl").exists())
            state = json.loads((root / MIGRATION_STATE_FILE).read_text(encoding="utf-8"))
            self.assertEqual(int(state.get("schema_version") or 0), 2)

            second = run_auto_migration(
                memory_dir=root,
                target_schema_version=2,
                auto_on_start=True,
                facts_scope="user_only",
            )
            self.assertFalse(bool(second.get("ran")))
            self.assertEqual(str(second.get("reason") or ""), "up_to_date")

    def test_migration_removes_assistant_noise_and_keeps_confirmed_user_facts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_auto_migrate_noise_") as tmpdir:
            root = Path(tmpdir)
            self._seed_migration_fixture(root)
            out = run_auto_migration(
                memory_dir=root,
                target_schema_version=2,
                auto_on_start=True,
                facts_scope="user_only",
            )
            self.assertTrue(bool(out.get("success")))

            assistant_profile = json.loads((root / "assistant_profile_store.json").read_text(encoding="utf-8"))
            self.assertFalse(dict(assistant_profile.get("profiles") or {}))

            vector_payload = json.loads((root / "vector_store.json").read_text(encoding="utf-8"))
            records = [dict(x) for x in list(vector_payload.get("records") or []) if isinstance(x, dict)]
            fact_rows = [x for x in records if str(dict(x.get("metadata") or {}).get("type") or "") == "fact"]
            self.assertTrue(any(str(x.get("text") or "").startswith("user.birth_year=2003") for x in fact_rows))
            self.assertFalse(any("assistant.birth_year" in str(x.get("text") or "") for x in fact_rows))
            statuses = {str(dict(x.get("metadata") or {}).get("status") or "").strip() for x in fact_rows}
            self.assertTrue(statuses.issubset({"confirmed", "confirmed_by_user"}))

    def _seed_migration_fixture(self, root: Path) -> None:
        events = [
            {
                "event_id": "e1",
                "ts": "2026-03-05T10:00:00+02:00",
                "type": "user_message",
                "payload": {"role": "user", "text": "I was born in 2003", "metadata": {"lang": "en"}, "ids": {}},
                "tags": [],
            },
            {
                "event_id": "e2",
                "ts": "2026-03-05T10:00:01+02:00",
                "type": "user_message",
                "payload": {"role": "user", "text": "I was born in 2003", "metadata": {"lang": "en"}, "ids": {}},
                "tags": [],
            },
            {
                "event_id": "e3",
                "ts": "2026-03-05T10:00:02+02:00",
                "type": "assistant_message",
                "payload": {
                    "role": "assistant",
                    "text": "[PARAMETERS]\nname=Luna\n\n[RESPONSE]\nI can help.",
                    "metadata": {"lang": "en"},
                    "ids": {},
                },
                "tags": [],
            },
        ]
        _write_jsonl(root / "events.jsonl", events)
        _write_json(
            root / "short_memory.json",
            {
                "items": [
                    {"id": "e1", "role": "user", "type": "message", "text": "I was born in 2003", "meta": {}},
                    {"id": "e3", "role": "assistant", "type": "message", "text": "[PARAMETERS]\na=1\n\n[RESPONSE]\nI can help.", "meta": {}},
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
                        "text": "[PARAMETERS]\na=1\n\n[RESPONSE]\nI can help.",
                        "thinking": "",
                        "created_at": 1.0,
                        "updated_at": "2026-03-05T10:00:03+02:00",
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
                        "updated_at": "2026-03-05T10:00:04+02:00",
                        "source": "fact",
                        "tags": ["fact", "subject_assistant", "key_birth_year"],
                        "importance": 0.9,
                        "confidence": 0.9,
                        "meta": {"fact": {"subject": "assistant", "key": "birth_year", "value": "1999"}},
                    },
                ]
            },
        )
        _write_json(
            root / "vector_store.json",
            {
                "records": [
                    {
                        "id": "msg:e3",
                        "text": "[PARAMETERS]\na=1\n\n[RESPONSE]\nI can help.",
                        "embedding": [0.5] + [0.0] * 31,
                        "metadata": {"type": "message", "role": "assistant"},
                    },
                    {
                        "id": "fact:doc-fact-old",
                        "text": "assistant.birth_year=1999",
                        "embedding": [0.4] + [0.0] * 31,
                        "metadata": {"type": "fact", "status": "confirmed", "source": "fact"},
                    },
                ]
            },
        )
        _write_json(root / "user_profile_store.json", {"profiles": {}, "versions": {}, "pending_facts": {}, "confirmed_facts": {}})
        _write_json(
            root / "assistant_profile_store.json",
            {
                "profiles": {"default": {"birth_year": {"value": "1999", "confidence": 0.9}}},
                "versions": {},
                "pending_facts": {},
                "confirmed_facts": {"default": {"birth_year": {"key": "birth_year", "value": "1999", "status": "confirmed"}}},
            },
        )
        _write_json(root / MIGRATION_STATE_FILE, {"schema_version": 1, "status": "legacy"})


if __name__ == "__main__":
    unittest.main()
