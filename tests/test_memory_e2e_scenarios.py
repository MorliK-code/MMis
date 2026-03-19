from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

from core.character_runtime import CharacterRuntime
from core.response_pipeline import (
    PROFILE_BALANCED,
    GenerateStage,
    PipelineContext,
    PostprocessStage,
    PromptBuildStage,
    PromptEngineStage,
)
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, ModelInfo, ProviderHealth
from llm.tokenizer import create_tokenizer
from memory.context_builder import ContextBuilderV2
from memory.dialog_episode_builder import DialogEpisodeBuilder, DialogTurn
from memory.dialog_episode_retriever import DialogEpisodeRetriever
from memory.document_memory import ChunkingConfig, DocumentMemory
from memory.document_retrieval import DocumentRetriever
from memory.fact_extractor import FactExtractor
from memory.governor import MemoryGovernor
from memory.long_memory import LongMemoryV2
from memory.memory_lifecycle import MemoryLifecycleManager
from memory.memory_manager import MemoryManager
from memory.memory_models import (
    ContextBuildRequest,
    DocumentIngestRequest,
    LifecycleDecision,
    MemoryEvent,
    MemoryLevel,
    MemoryRecord,
    MemoryScope,
    MemoryStatus,
    MemoryType,
)
from memory.memory_policy import MemoryPolicy
from memory.memory_scoring import SalienceWeights, ScoreWeights
from memory.reranker import HeuristicReranker
from memory.retrieval import HybridRetriever
from memory.retrieval_projection import record_search_text
from modules.nlu.normalizer import normalize_text
from prompt_engine.prompt_engine import PromptEngine


class _PromptAwareProvider(LLMProviderBase):
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def generate(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        system_prompt = "\n\n".join(str(msg.content or "") for msg in req.messages if msg.role == "system")
        user_message = ""
        for msg in reversed(list(req.messages or [])):
            if msg.role == "user":
                user_message = str(msg.content or "")
                break
        return LLMResponse(
            text=self._answer_for_prompt(system_prompt=system_prompt, user_message=user_message),
            model="fake-e2e-model",
        )

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake-e2e", detail="ok", model="fake-e2e-model")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="fake-e2e", model=model or "fake-e2e-model")

    def list_models(self) -> list[str]:
        return ["fake-e2e-model"]

    @classmethod
    def _answer_for_prompt(cls, *, system_prompt: str, user_message: str) -> str:
        user_norm = normalize_text(str(user_message or "")).lower()
        self_facts = cls._extract_block(system_prompt, "SELF_FACTS")
        claims = cls._extract_block(system_prompt, "RELEVANT_CLAIMS")
        recalled_dialog = cls._extract_block(system_prompt, "RECALLED_DIALOG")
        document_evidence = cls._extract_block(system_prompt, "DOCUMENT_EVIDENCE")
        supporting_messages = cls._extract_block(system_prompt, "SUPPORTING_MESSAGES")

        if "gpu" in user_norm or "videocard" in user_norm or "graphics" in user_norm:
            gpu = cls._fact_value(self_facts, "environment_gpu_model")
            if gpu:
                return f"Your GPU is {gpu}."
            return "I do not see the exact GPU fact in memory."

        if "what do i use" in user_norm or "what i use" in user_norm:
            match = re.search(r"-\s*user\s+uses\s+(.+)", claims, re.I)
            if match:
                return f"You use {match.group(1).strip()}."
            return "I do not see a confirmed use-claim in memory."

        if "what did we discuss" in user_norm or "discuss about memory" in user_norm:
            summary = cls._line_value(recalled_dialog, "Summary")
            decisions = cls._section_bullets(recalled_dialog, "Decisions:")
            if "assistant thoughts" in (recalled_dialog + "\n" + supporting_messages).lower() and decisions:
                return f"We discussed assistant thoughts in memory design and decided that {decisions[0]}."
            if summary and decisions:
                return f"{summary} We decided that {decisions[0]}."
            if summary:
                return summary
            return "I do not see a recalled dialog episode for that topic."

        if "where in code" in user_norm or "ollama" in user_norm:
            title = cls._line_value(document_evidence, "Document")
            function_name = cls._document_function_name(document_evidence)
            chunk_text = document_evidence
            if title and chunk_text:
                function_name = function_name or "the shown function"
                return f"In {title}, ollama is called in {function_name} via ollama.chat(...)."
            return "I do not see matching document evidence for that code question."

        return "I do not have a scripted answer for this test."

    @staticmethod
    def _extract_block(prompt: str, label: str) -> str:
        lines = str(prompt or "").splitlines()
        header = f"[{label}]"
        capture = False
        out: list[str] = []
        for raw_line in lines:
            line = str(raw_line or "")
            stripped = line.strip()
            if stripped == header:
                capture = True
                continue
            if capture and re.match(r"^\[[A-Z_]+\]$", stripped):
                break
            if capture:
                out.append(line.rstrip())
        return "\n".join(out).strip()

    @staticmethod
    def _fact_value(block: str, predicate: str) -> str:
        match = re.search(rf"-\s*{re.escape(predicate)}:\s*(.+)", str(block or ""), re.I)
        return str(match.group(1) if match else "").strip()

    @staticmethod
    def _line_value(block: str, label: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.+)$", str(block or ""), re.I | re.M)
        return str(match.group(1) if match else "").strip()

    @staticmethod
    def _section_bullets(block: str, header: str) -> list[str]:
        lines = str(block or "").splitlines()
        capture = False
        out: list[str] = []
        for raw_line in lines:
            stripped = str(raw_line or "").strip()
            if stripped == str(header or "").strip():
                capture = True
                continue
            if capture and stripped.endswith(":") and not stripped.startswith("-"):
                break
            if capture and stripped.startswith("-"):
                out.append(stripped.removeprefix("-").strip())
        return out

    @staticmethod
    def _first_chunk_text(block: str) -> str:
        match = re.search(r"^Chunk(?:\s+\d+)?:\s*(.+)$", str(block or ""), re.I | re.M)
        return str(match.group(1) if match else "").strip()

    @staticmethod
    def _document_function_name(block: str) -> str:
        match = re.search(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(block or ""))
        return str(match.group(1) if match else "").strip()


def _record(
    record_id: str,
    text: str,
    *,
    namespace: str,
    memory_type: MemoryType,
    level: MemoryLevel,
    metadata: dict | None = None,
    scope: MemoryScope = MemoryScope.CONVERSATION,
    source_event_id: str = "",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    importance: float = 0.72,
    confidence: float = 0.84,
    created_at: float | None = None,
) -> MemoryRecord:
    now = float(created_at or time.time())
    return MemoryRecord(
        id=record_id,
        text=text,
        memory_type=memory_type,
        level=level,
        scope=scope,
        namespace=namespace,
        metadata=dict(metadata or {}),
        importance=importance,
        confidence=confidence,
        created_at=now,
        updated_at=now,
        status=status,
        source_event_id=source_event_id,
    )


class _E2EStore:
    def __init__(self) -> None:
        self.reindex_required = False
        self._rows: dict[str, MemoryRecord] = {}
        self._order: list[str] = []

    def upsert(self, record: MemoryRecord) -> None:
        key = str(record.id)
        if key not in self._rows:
            self._order.append(key)
        self._rows[key] = record

    def batch_upsert(self, records: list[MemoryRecord]) -> None:
        for record in list(records or []):
            self.upsert(record)

    def iter_records(self, namespace: str | None = None) -> list[MemoryRecord]:
        rows = [self._rows[key] for key in list(self._order)]
        if namespace is None:
            return rows
        return [row for row in rows if str(row.namespace or "") == str(namespace or "")]

    def children_of(self, parent_id: str, namespace: str | None = None, include_stale: bool = True) -> list[MemoryRecord]:
        rows = [
            row
            for row in self.iter_records(namespace=namespace)
            if str(row.parent_id or "") == str(parent_id or "")
        ]
        if include_stale:
            return rows
        return [row for row in rows if row.status == MemoryStatus.ACTIVE]

    def search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        hits = self.lexical_search(
            query_text=query_text,
            top_k=top_k,
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=metadata_filters,
        )
        return [record.to_dict() for record, _score in list(hits or [])]

    def semantic_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        return self._scored_hits(
            query_text=str(query_text or ""),
            top_k=top_k,
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=metadata_filters,
            channel="semantic",
        )

    def lexical_search(self, *, query_text, top_k, namespace, scopes, include_stale, metadata_filters):
        return self._scored_hits(
            query_text=str(query_text or ""),
            top_k=top_k,
            namespace=namespace,
            scopes=scopes,
            include_stale=include_stale,
            metadata_filters=metadata_filters,
            channel="lexical",
        )

    def close(self) -> None:
        return None

    def _scored_hits(
        self,
        *,
        query_text: str,
        top_k: int,
        namespace: str,
        scopes: list[MemoryScope],
        include_stale: bool,
        metadata_filters: dict | None,
        channel: str,
    ) -> list[tuple[MemoryRecord, float]]:
        out: list[tuple[MemoryRecord, float]] = []
        for record in self.iter_records(namespace=namespace):
            if scopes and record.scope not in list(scopes or []):
                continue
            if not include_stale and record.status not in {MemoryStatus.ACTIVE, MemoryStatus.SUPERSEDED}:
                continue
            if not self._matches_filters(record, dict(metadata_filters or {})):
                continue
            score = self._score(query_text=query_text, record=record, channel=channel)
            out.append((record, score))
        out.sort(key=lambda item: float(item[1]), reverse=True)
        return out[: max(1, int(top_k or 1))]

    def _matches_filters(self, record: MemoryRecord, filters: dict) -> bool:
        for key, value in dict(filters or {}).items():
            if key == "memory_type":
                allowed = {str(x).strip().lower() for x in list(value or []) if str(x).strip()}
                if allowed and str(record.memory_type.value).strip().lower() not in allowed:
                    return False
                continue
            if key.startswith("metadata."):
                actual = self._metadata_path(record.metadata, key.removeprefix("metadata."))
                if str(actual or "").strip().lower() != str(value or "").strip().lower():
                    return False
        return True

    @staticmethod
    def _metadata_path(metadata: dict | None, path: str):
        current = dict(metadata or {})
        for part in [str(x).strip() for x in str(path or "").split(".") if str(x).strip()]:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    @staticmethod
    def _score(*, query_text: str, record: MemoryRecord, channel: str) -> float:
        query_norm = normalize_text(str(query_text or "")).lower()
        record_norm = normalize_text(record_search_text(record.text, metadata=record.metadata)).lower()
        query_tokens = {token for token in query_norm.split() if token}
        record_tokens = {token for token in record_norm.split() if token}
        overlap = len(query_tokens.intersection(record_tokens))
        baseline = 0.34 if channel == "lexical" else 0.32
        if overlap:
            baseline += min(0.38, 0.12 * overlap)
        if query_norm and query_norm in record_norm:
            baseline += 0.16
        if record.memory_type == MemoryType.CLAIM:
            baseline += 0.03
        if record.memory_type == MemoryType.EPISODE:
            baseline += 0.02
        if record.memory_type in {MemoryType.DOCUMENT, MemoryType.DOCUMENT_CHUNK}:
            baseline += 0.02
        return min(0.96, baseline)


class _E2EEventStore:
    def append(self, row) -> None:  # noqa: ANN001
        _ = row


class _E2EDebugger:
    def record_retrieval_trace(self, **kwargs) -> None:  # noqa: ANN003
        _ = kwargs

    def snapshot(self, request):  # noqa: ANN001
        _ = request
        return {}


class _E2ELifecycle:
    def __init__(self) -> None:
        self._delegate = MemoryLifecycleManager()

    def decide(self, record: MemoryRecord, *, now_ts: float) -> LifecycleDecision:
        _ = (record, now_ts)
        return LifecycleDecision(reason="e2e_test_no_promotion", route="e2e")

    def resolve_conflict(self, *, old: MemoryRecord, new: MemoryRecord):
        return self._delegate.resolve_conflict(old=old, new=new)


def _manager() -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._cfg = SimpleNamespace()
    temp_dir = Path(tempfile.mkdtemp(prefix="mmis-memory-e2e-"))
    manager._root = temp_dir
    manager._state_path = temp_dir / "manager_state.json"
    manager._lock = RLock()
    manager._storage_profile = "compact"
    manager._DEBUG_METADATA_FIELDS = {
        "web_used", "web_factual_mode", "web_query_intent", "web_search_mode",
        "web_primary_category", "web_evidence_quality_score", "web_evidence_quality",
        "web_sources_scanned", "web_sources_selected", "web_evidence_count",
        "web_conflicting_sources", "web_low_evidence_quality", "web_selected_avg_quality",
        "web_topical_filtered_sources", "web_conflict_severity", "web_evidence_strength",
        "web_final_factual_confidence", "web_cautious_synthesis",
        "web_numeric_candidates_selected", "web_numeric_candidates_rejected",
        "_persona_snapshot_debug", "persona_snapshot",
        "promotion_project_signal", "promotion_task_signal", "promotion_decision_signal",
        "promotion_preference_signal", "promotion_issue_signal", "promotion_technical_signal",
        "promotion_repeated_topic_signal", "promotion_smalltalk_signal",
        "promotion_signal_score", "promotion_stable_fact_signal", "promotion_fact_signal",
        "promotion_fact_relation_diversity",
        "extracted_facts_count", "extracted_fact_relations",
        "memory_analysis", "memory_views_debug", "claim_candidates",
        "lifecycle_decision", "decision_debug",
        "assistant_write_policy", "assistant_write_blocked", "assistant_write_reason",
    }
    manager._store = _E2EStore()
    manager._event_store = _E2EEventStore()
    manager._fact_extractor = FactExtractor()
    manager._policy = MemoryPolicy()
    manager._lifecycle = _E2ELifecycle()
    manager._governor = MemoryGovernor(lifecycle=manager._lifecycle)
    manager._retriever = HybridRetriever(
        store=manager._store,
        stale_after_days=30,
        weights=ScoreWeights(),
    )
    manager._dialog_episode_retriever = DialogEpisodeRetriever(store=manager._store)
    manager._document_retriever = DocumentRetriever(store=manager._store)
    manager._reranker = HeuristicReranker()
    manager._context_builder = ContextBuilderV2(tokenizer=create_tokenizer())
    manager._document_memory = DocumentMemory(
        store=manager._store,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=60),
    )
    manager._long_memory = LongMemoryV2(store=manager._store, document_memory=manager._document_memory)
    manager._debugger = _E2EDebugger()
    manager._context_compressor = None
    manager._working_records = []
    manager._session_summary = ""
    manager._open_questions = []
    manager._current_decisions = []
    manager._active_preferences = []
    manager._private_runtime = {}
    manager._temporary_ttl_sec = 3600
    manager._private_runtime_ttl_sec = 900
    manager._working_limit = 120
    manager._retrieval_top_k = 8
    manager._rerank_top_k = 8
    manager._importance_weights = {
        "base": 0.42,
        "decision": 0.24,
        "remember": 0.18,
        "project": 0.10,
    }
    manager._salience_weights = SalienceWeights(
        novelty=0.22,
        permanence=0.20,
        repetition=0.14,
        project_relevance=0.16,
        task_relevance=0.16,
        explicit_save_signal=0.12,
    )
    manager._save_state = lambda: None
    manager._cleanup_expired = lambda: None
    return manager


def _ingest_message(manager: MemoryManager, *, namespace: str, role: str, text: str) -> None:
    manager.ingest_event(
        MemoryEvent(
            role=role,
            text=text,
            namespace=namespace,
            scope=MemoryScope.CONVERSATION,
            memory_type=MemoryType.MESSAGE,
            metadata={"conversation_id": namespace},
        )
    )


def _build_context_and_prompt(
    manager: MemoryManager,
    *,
    namespace: str,
    user_message: str,
    top_k: int = 6,
) -> tuple[object, PipelineContext]:
    result = manager.build_context(
        ContextBuildRequest(
            system_prompt="",
            user_message=user_message,
            namespace=namespace,
            scopes=[MemoryScope.CONVERSATION, MemoryScope.PROJECT, MemoryScope.SESSION, MemoryScope.GLOBAL_USER],
            top_k=top_k,
        )
    )
    pack = result.to_dict()
    ctx = PipelineContext(
        route="chat",
        user_msg=user_message,
        clean_user_msg=user_message,
        state={},
        meta={},
        tags={},
        retrieved_memories=list(pack.get("selected") or []),
        traits={},
        policies={},
        profile=PROFILE_BALANCED,
        memory_context=pack,
    )
    build_stage = PromptBuildStage(character_runtime=CharacterRuntime())
    engine_stage = PromptEngineStage(prompt_engine=PromptEngine())
    ctx = build_stage.run(ctx)
    ctx = engine_stage.run(ctx)
    return result, ctx


def _system_prompt(ctx: PipelineContext) -> str:
    return str(ctx.prompt_sections.get("system") or "")


def _generate_answer(ctx: PipelineContext) -> tuple[PipelineContext, _PromptAwareProvider]:
    provider = _PromptAwareProvider()
    generate_stage = GenerateStage(provider=provider, character_runtime=CharacterRuntime())
    postprocess_stage = PostprocessStage()
    ctx = generate_stage.run(ctx)
    ctx = postprocess_stage.run(ctx)
    return ctx, provider


def test_golden_e2e_fact_recall_survives_assistant_noise() -> None:
    manager = _manager()
    namespace = "golden-fact"

    _ingest_message(manager, namespace=namespace, role="user", text="My name is Pasha.")
    _ingest_message(manager, namespace=namespace, role="user", text="I use RTX 3050 Ti with 32 GB RAM.")

    noise = _record(
        "message:noise-assistant",
        "I don't remember your GPU. Check it yourself in Device Manager.",
        namespace=namespace,
        memory_type=MemoryType.MESSAGE,
        level=MemoryLevel.L2_EPISODIC,
        metadata={"source_kind": "assistant_reply", "assistant_memory_help_noise": True},
    )
    manager._store.upsert(noise)

    result, ctx = _build_context_and_prompt(
        manager,
        namespace=namespace,
        user_message="What is my GPU?",
        top_k=4,
    )

    assert result.recall_mode == "exact_fact_recall"
    assert "self_facts" in result.blocks
    assert "environment_gpu_model: RTX 3050 Ti" in str(result.blocks["self_facts"])
    system_prompt = _system_prompt(ctx)
    assert "[SELF_FACTS]" in system_prompt
    assert "environment_gpu_model: RTX 3050 Ti" in system_prompt
    assert "Device Manager" not in system_prompt
    assert "Check it yourself" not in system_prompt
    assert "don't remember your GPU" not in system_prompt.lower()
    assert str(ctx.tags.get("self_memory_exact") or "").lower() == "true"
    ctx, provider = _generate_answer(ctx)
    assert len(provider.requests) == 1
    assert ctx.text == "Your GPU is RTX 3050 Ti."
    assert "device manager" not in ctx.text.lower()
    assert "check it yourself" not in ctx.text.lower()


def test_golden_e2e_claim_recall_promotes_and_surfaces_relevant_claims() -> None:
    manager = _manager()
    namespace = "golden-claim"

    _ingest_message(manager, namespace=namespace, role="user", text="I use VS Code.")
    _ingest_message(manager, namespace=namespace, role="user", text="I have a Samsung fridge.")

    result, ctx = _build_context_and_prompt(
        manager,
        namespace=namespace,
        user_message="What do I use?",
        top_k=1,
    )

    claim_rows = [
        row for row in manager._store.iter_records(namespace=namespace)
        if row.memory_type == MemoryType.CLAIM
    ]
    assert any(
        str(dict(row.metadata or {}).get("claim", {}).get("predicate") or "").strip().lower() == "uses"
        for row in claim_rows
    )
    assert "relevant_claims" in result.blocks
    assert "- user uses VS Code" in str(result.blocks["relevant_claims"])
    assert "Samsung fridge" not in str(result.blocks["relevant_claims"])
    system_prompt = _system_prompt(ctx)
    assert "[RELEVANT_CLAIMS]\n- user uses VS Code" in system_prompt
    ctx, provider = _generate_answer(ctx)
    assert len(provider.requests) == 1
    assert ctx.text == "You use VS Code."
    assert "Samsung fridge" not in ctx.text


def test_golden_e2e_dialog_episode_recall_builds_episode_and_prompt_block() -> None:
    manager = _manager()
    namespace = "golden-episode"

    turns_source: list[tuple[str, str]] = [
        ("user", "We should not store assistant thoughts in long-term memory."),
        ("assistant", "Right, assistant thoughts would pollute retrieval."),
        ("user", "Let's keep them only in debug logs."),
        ("assistant", "We decided to keep assistant thoughts in debug only and move on."),
    ]
    built_turns: list[DialogTurn] = []
    for idx, (role, text) in enumerate(turns_source):
        _ingest_message(manager, namespace=namespace, role=role, text=text)
        row = next(
            record
            for record in reversed(list(manager._store.iter_records(namespace=namespace)))
            if record.memory_type == MemoryType.MESSAGE and str(record.text or "") == text
        )
        built_turns.append(
            DialogTurn(
                turn_id=str(row.id),
                role=role,
                text=text,
                ts=float(row.created_at or time.time()) + idx,
                topic="memory design",
                metadata=dict(row.metadata or {}),
            )
        )

    builder = DialogEpisodeBuilder()
    episode = builder.build_episode(built_turns, episode_id="episode:memory-design")
    assert episode is not None
    manager._store.upsert(
        _record(
            str(episode.id),
            str(episode.summary_short),
            namespace=namespace,
            memory_type=MemoryType.EPISODE,
            level=MemoryLevel.L2_EPISODIC,
            metadata={
                "dialog_episode": episode.to_dict(),
                "topic": episode.topic,
                "summary_short": episode.summary_short,
                "summary_reasoning": episode.summary_reasoning,
                "turn_ids": list(episode.turn_ids or []),
                "decisions": list(episode.decisions or []),
                "open_questions": list(episode.open_questions or []),
                "participants": list(episode.participants or []),
                "topic_keys": list(episode.topic_keys or []),
                "entity_keys": list(episode.entity_keys or []),
                "salience": float(episode.salience or 0.0),
            },
        )
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace=namespace,
        user_message="What did we discuss about memory?",
        top_k=4,
    )

    assert result.recall_mode == "contextual_recall"
    assert "recalled_dialog" in result.blocks
    assert "Reasoning:" in str(result.blocks["recalled_dialog"])
    assert "memory design" in str(result.blocks["recalled_dialog"]).lower()
    assert "debug logs" in str(result.blocks["recalled_dialog"]).lower()
    system_prompt = _system_prompt(ctx)
    prompt_memory_text = str(dict(ctx.state.get("prompt_memory_block") or {}).get("text") or "")
    assert "[RECALLED_DIALOG]" in (system_prompt + "\n" + prompt_memory_text)
    assert "assistant thoughts" in (system_prompt + "\n" + prompt_memory_text).lower()
    assert "[SUPPORTING_MESSAGES]" in (system_prompt + "\n" + prompt_memory_text)
    assert "[EPISODIC_MEMORIES]" not in (system_prompt + "\n" + prompt_memory_text)


def test_golden_e2e_dialog_episode_recall_uses_episode_anchors_when_summary_is_weak() -> None:
    manager = _manager()
    namespace = "golden-episode-anchor"

    manager._store.upsert(
        _record(
            "episode:memory-plan-anchor",
            "Discussed this.",
            namespace=namespace,
            memory_type=MemoryType.EPISODE,
            level=MemoryLevel.L2_EPISODIC,
            metadata={
                "dialog_episode": {
                    "id": "episode:memory-plan-anchor",
                    "topic": "general_dialog",
                    "turn_ids": ["evt:a1", "evt:a2"],
                    "summary_short": "Discussed this.",
                    "summary_reasoning": "We aligned the memory roadmap around facts, claims, and dialog episodes.",
                    "decisions": ["First stabilize memory, then return to web."],
                    "open_questions": ["How should soft singleton groups work?"],
                    "participants": ["user", "assistant"],
                    "salience": 0.9,
                    "topic_keys": ["memory", "plan"],
                    "entity_keys": ["soft_singleton", "memory_governor"],
                    "created_at": 100.0,
                    "updated_at": 120.0,
                },
                "topic": "general_dialog",
                "summary_short": "Discussed this.",
                "summary_reasoning": "We aligned the memory roadmap around facts, claims, and dialog episodes.",
                "turn_ids": ["evt:a1", "evt:a2"],
                "decisions": ["First stabilize memory, then return to web."],
                "open_questions": ["How should soft singleton groups work?"],
                "participants": ["user", "assistant"],
                "topic_keys": ["memory", "plan"],
                "entity_keys": ["soft_singleton", "memory_governor"],
                "salience": 0.9,
            },
        )
    )
    _ingest_message(manager, namespace=namespace, role="user", text="ok")
    _ingest_message(manager, namespace=namespace, role="assistant", text="We touched memory a bit.")

    result, ctx = _build_context_and_prompt(
        manager,
        namespace=namespace,
        user_message="What was our plan for memory?",
        top_k=4,
    )

    assert result.recall_mode == "contextual_recall"
    assert "recalled_dialog" in result.blocks
    block = str(result.blocks["recalled_dialog"])
    assert "First stabilize memory, then return to web." in block
    assert "soft singleton groups" in block.lower()
    system_prompt = _system_prompt(ctx)
    prompt_memory_text = str(dict(ctx.state.get("prompt_memory_block") or {}).get("text") or "")
    assert "[RECALLED_DIALOG]" in (system_prompt + "\n" + prompt_memory_text)
    assert "[EPISODIC_MEMORIES]" not in (system_prompt + "\n" + prompt_memory_text)


def test_golden_e2e_document_recall_surfaces_document_evidence() -> None:
    manager = _manager()
    namespace = "golden-doc"

    manager.ingest_document(
        DocumentIngestRequest(
            text=(
                "# LLM helpers\n"
                "def call_ollama(prompt):\n"
                "    return ollama.chat(model='qwen', messages=[{'role': 'user', 'content': prompt}])\n\n"
                "# Other code\n"
                "class Settings:\n"
                "    pass\n"
            ),
            source="memory/llm_helpers.py",
            namespace=namespace,
            scope=MemoryScope.PROJECT,
            title="llm_helpers.py",
            metadata={"path": "memory/llm_helpers.py"},
        )
    )

    result, ctx = _build_context_and_prompt(
        manager,
        namespace=namespace,
        user_message="Where in code is ollama called?",
        top_k=4,
    )

    assert "document_evidence" in result.blocks
    block = str(result.blocks["document_evidence"])
    assert "Document: llm_helpers.py" in block
    assert "ollama.chat" in block
    system_prompt = _system_prompt(ctx)
    assert "[DOCUMENT_EVIDENCE]" in system_prompt
    assert "ollama.chat" in system_prompt
    ctx, provider = _generate_answer(ctx)
    assert len(provider.requests) == 1
    assert "llm_helpers.py" in ctx.text
    assert "call_ollama" in ctx.text
    assert "ollama.chat" in ctx.text
