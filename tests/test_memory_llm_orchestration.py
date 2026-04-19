from __future__ import annotations

import json
import time
import threading
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import memory_core.adapter as adapter_module
import llm.ollama_provider as ollama_provider_module
from llm.ollama_provider import OllamaProvider, _memory_llm_interrupt
from llm.priority_manager import LLMPriorityManager
from llm.provider_base import LLMRequest, LLMResponse, Message, ModelInfo, ProviderHealth
from llm.task_models import TaskModelProfile, TaskModelRegistry
from llm.task_router import TaskModelRouter
from memory_core.adapter import MemoryCoreAdapter, _memory_llm_lock
from memory_core.facade import MemoryService
from memory_core.processors.memory_llm_processor import MemoryLLMProcessor
from memory_core.storage.job_queue_store import IngestJob, JobQueueStore
from memory_core.schemas import MemoryEnvelope
from memory_core.storage.sqlite_db import Database
from memory_core.worker.background_worker import BackgroundWorker, WorkerConfig


class _FakeControlledProvider:
    def __init__(self) -> None:
        self.pause_calls = 0
        self.resume_calls = 0
        self.shutdown_calls = 0
        self.warmup_calls: list[object] = []

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def warmup(self, keep_alive=None) -> bool:
        self.warmup_calls.append(keep_alive)
        return True


class _FakeTaskRouter:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.provider = _FakeControlledProvider()

    def run_task_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            text=json.dumps(
                {
                    "event_id": "fake-event",
                    "importance": 0.7,
                    "should_process": True,
                    "proposals": [],
                },
                ensure_ascii=False,
            )
        )

    def get_cached_provider(self, task_name: str):
        if task_name == MemoryLLMProcessor.TASK_NAME:
            return self.provider
        return None

    def shutdown_cached_provider(self, task_name: str) -> None:
        if task_name == MemoryLLMProcessor.TASK_NAME:
            self.provider.shutdown()


class _JsonTaskRouter(_FakeTaskRouter):
    def __init__(self) -> None:
        super().__init__()
        self.json_calls: list[dict] = []

    def run_task_model_json(self, **kwargs):
        self.json_calls.append(dict(kwargs))
        return SimpleNamespace(
            text=json.dumps(
                {
                    "event_id": "fake-event",
                    "importance": 0.7,
                    "should_process": True,
                    "proposals": [],
                },
                ensure_ascii=False,
            )
        )

    def run_task_model(self, **kwargs):
        raise AssertionError("run_task_model should not be used when run_task_model_json is available")


class _SlowTaskRouter(_FakeTaskRouter):
    def __init__(self, delay_sec: float) -> None:
        super().__init__()
        self.delay_sec = float(delay_sec)

    def run_task_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        time.sleep(self.delay_sec)
        return super().run_task_model(**kwargs)


class _ErrorTaskRouter(_FakeTaskRouter):
    def __init__(self, message: str = "provider unavailable", *, set_interrupt: bool = False) -> None:
        super().__init__()
        self.message = message
        self.set_interrupt = set_interrupt

    def run_task_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        if self.set_interrupt:
            _memory_llm_interrupt.set()
        raise RuntimeError(self.message)


class _InvalidJsonTaskRouter(_FakeTaskRouter):
    def __init__(self, text: str = "this is not valid json") -> None:
        super().__init__()
        self.text = text

    def run_task_model(self, **kwargs):
        self.calls.append(dict(kwargs))
        return SimpleNamespace(text=self.text)


class _FakeStreamingClient:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.chat_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    def chat(self, **kwargs):
        self.chat_calls.append(dict(kwargs))
        result = self._chunks() if callable(self._chunks) else self._chunks
        return iter(result)

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        return {}


class _FakeWorker:
    def __init__(self, running: bool) -> None:
        self.running = running
        self.start_calls = 0
        self.resume_calls = 0
        self.wake_calls = 0

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.start_calls += 1
        self.running = True

    def resume(self) -> None:
        self.resume_calls += 1

    def wake(self) -> None:
        self.wake_calls += 1


class _FakeThread:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _FakeClosableClient:
    def __init__(self) -> None:
        self.close_calls = 0
        self.transport_close_calls = 0
        self.transport = SimpleNamespace(close=self._close_transport)

    def close(self) -> None:
        self.close_calls += 1

    def _close_transport(self) -> None:
        self.transport_close_calls += 1


class _FakePriorityManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def wait_for_turn(self, priority: int, timeout: float | None = None) -> bool:
        self.calls.append(("wait", priority))
        return True

    def release(self, priority: int) -> None:
        self.calls.append(("release", priority))


class _ScriptedProvider:
    def __init__(self, scripted) -> None:
        self._scripted = scripted

    def generate(self, req: LLMRequest) -> LLMResponse:
        if not self._scripted:
            raise RuntimeError("No scripted response")
        action = self._scripted.pop(0)
        if isinstance(action, Exception):
            raise action
        if isinstance(action, LLMResponse):
            return action
        return LLMResponse(text=str(action), model=str(req.model or "fake-memory-model"))

    def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake", detail="ok", model="fake-memory-model")

    def model_info(self, model: str = "") -> ModelInfo:
        return ModelInfo(provider="fake", model=model or "fake-memory-model")

    def list_models(self) -> list[str]:
        return ["fake-memory-model"]


class _ImmediateRequeueQueue:
    def __init__(self, job) -> None:
        self._job = job
        self.requeue_calls: list[tuple[str, str]] = []
        self.complete_calls: list[str] = []
        self.fail_calls: list[tuple[str, str, bool]] = []

    def dequeue(self, **kwargs):
        job, self._job = self._job, None
        return job

    def requeue_immediately(self, job_id: str, error_text: str = "") -> bool:
        self.requeue_calls.append((job_id, error_text))
        return True

    def complete(self, job_id: str) -> bool:
        self.complete_calls.append(job_id)
        return True

    def fail(self, job_id: str, error_text: str = "", retry: bool = True) -> bool:
        self.fail_calls.append((job_id, error_text, retry))
        return retry


class _EventStoreForInterrupt:
    def __init__(self, event) -> None:
        self._event = event

    def get_by_id(self, event_id: str):
        return self._event


class MemoryLLMOrchestrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        _memory_llm_interrupt.clear()

    def test_memory_processor_marks_requests_as_memory_and_uses_configured_keep_alive(self) -> None:
        router = _FakeTaskRouter()
        processor = MemoryLLMProcessor(task_router=router, keep_alive="30m")

        processor.process(
            MemoryEnvelope(
                source_kind="user",
                payload_type="message",
                text="Пользователь работает над Python проектом.",
            )
        )

        self.assertEqual(len(router.calls), 1)
        call = dict(router.calls[0])
        self.assertEqual(call["task_name"], MemoryLLMProcessor.TASK_NAME)
        metadata = dict(call.get("metadata") or {})
        self.assertEqual(metadata.get("source"), MemoryLLMProcessor.TASK_NAME)
        self.assertEqual(metadata.get("think"), False)
        self.assertEqual(metadata.get("keep_alive"), "30m")

    def test_memory_processor_does_not_force_zero_keep_alive_by_default(self) -> None:
        router = _FakeTaskRouter()
        processor = MemoryLLMProcessor(task_router=router)

        processor.process(
            MemoryEnvelope(
                source_kind="user",
                payload_type="message",
                text="Пользователь работает над Python проектом.",
            )
        )

        metadata = dict(router.calls[0].get("metadata") or {})
        self.assertEqual(metadata.get("source"), MemoryLLMProcessor.TASK_NAME)
        self.assertEqual(metadata.get("think"), False)
        self.assertNotIn("keep_alive", metadata)

    def test_memory_processor_prefers_json_task_model_and_disables_fallback(self) -> None:
        router = _JsonTaskRouter()
        processor = MemoryLLMProcessor(task_router=router, keep_alive="30m")

        processor.process(
            MemoryEnvelope(
                source_kind="assistant",
                payload_type="message",
                text="Пользователь просил запомнить рабочий контекст.",
            )
        )

        self.assertEqual(len(router.json_calls), 1)
        call = dict(router.json_calls[0])
        self.assertEqual(call["task_name"], MemoryLLMProcessor.TASK_NAME)
        self.assertEqual(tuple(call.get("required_fields") or ()), ("event_id", "importance", "should_process"))
        self.assertEqual(call.get("allow_array"), False)
        self.assertEqual(call.get("allow_fallback"), False)
        metadata = dict(call.get("metadata") or {})
        self.assertEqual(metadata.get("source"), MemoryLLMProcessor.TASK_NAME)
        self.assertEqual(metadata.get("think"), False)
        self.assertEqual(metadata.get("keep_alive"), "30m")

    def test_memory_processor_uses_empty_result_fallback_after_validation_failure(self) -> None:
        registry = TaskModelRegistry(
            {
                MemoryLLMProcessor.TASK_NAME: TaskModelProfile(
                    name=MemoryLLMProcessor.TASK_NAME,
                    provider="ollama",
                    model="fake-memory-model",
                    temperature=0.1,
                    max_tokens=256,
                    timeout=5.0,
                    enabled=True,
                )
            }
        )
        scripted = {MemoryLLMProcessor.TASK_NAME: ["   "]}

        def _provider_factory(profile):
            return _ScriptedProvider(scripted.setdefault(profile.name, []))

        processor = MemoryLLMProcessor(
            task_router=TaskModelRouter(registry=registry, provider_factory=_provider_factory)
        )

        result = processor.process(
            MemoryEnvelope(
                source_kind="user",
                payload_type="message",
                text="remember that the background task can safely skip empty outputs",
            )
        )

        self.assertFalse(result.should_process)
        self.assertEqual(result.proposals, [])
        self.assertEqual(result.importance, 0.0)
        self.assertIn('"proposals": []', result.raw_response)

    def test_memory_processor_treats_missing_proposals_field_as_empty_list(self) -> None:
        registry = TaskModelRegistry(
            {
                MemoryLLMProcessor.TASK_NAME: TaskModelProfile(
                    name=MemoryLLMProcessor.TASK_NAME,
                    provider="ollama",
                    model="fake-memory-model",
                    temperature=0.1,
                    max_tokens=256,
                    timeout=5.0,
                    enabled=True,
                )
            }
        )
        scripted = {
            MemoryLLMProcessor.TASK_NAME: [
                json.dumps(
                    {
                        "event_id": "fake-event",
                        "importance": 0.25,
                        "should_process": True,
                    },
                    ensure_ascii=False,
                )
            ]
        }

        def _provider_factory(profile):
            return _ScriptedProvider(scripted.setdefault(profile.name, []))

        processor = MemoryLLMProcessor(
            task_router=TaskModelRouter(registry=registry, provider_factory=_provider_factory)
        )

        result = processor.process(
            MemoryEnvelope(
                source_kind="user",
                payload_type="message",
                text="hello there",
            )
        )

        self.assertTrue(result.should_process)
        self.assertEqual(result.proposals, [])
        self.assertEqual(result.importance, 0.25)

    def test_memory_processor_hard_timeout_raises_and_resets_stuck_provider(self) -> None:
        router = _SlowTaskRouter(delay_sec=0.2)
        processor = MemoryLLMProcessor(task_router=router, timeout_sec=0.05)

        with self.assertRaises(TimeoutError):
            processor.process(
                MemoryEnvelope(
                    source_kind="user",
                    payload_type="message",
                    text="very short message",
                )
            )

        self.assertEqual(router.provider.shutdown_calls, 1)

    def test_memory_processor_raises_interrupted_error_when_main_lock_is_held(self) -> None:
        processor = MemoryLLMProcessor(task_router=_FakeTaskRouter())
        envelope = MemoryEnvelope(
            source_kind="user",
            payload_type="message",
            text="remember this after the response",
        )

        acquired = _memory_llm_lock.acquire(timeout=0.1)
        self.assertTrue(acquired)
        try:
            with self.assertRaises(InterruptedError):
                processor.process(envelope)
        finally:
            if acquired and _memory_llm_lock.locked():
                _memory_llm_lock.release()

    def test_memory_processor_raises_interrupted_error_when_interrupt_flag_is_set(self) -> None:
        processor = MemoryLLMProcessor(task_router=_FakeTaskRouter())
        envelope = MemoryEnvelope(
            source_kind="user",
            payload_type="message",
            text="remember this after the response",
        )

        _memory_llm_interrupt.set()

        with self.assertRaises(InterruptedError):
            processor.process(envelope)

    def test_memory_processor_detects_interrupt_epoch_even_after_flag_clears(self) -> None:
        processor = MemoryLLMProcessor(task_router=_SlowTaskRouter(delay_sec=0.3), timeout_sec=1.0)
        envelope = MemoryEnvelope(
            source_kind="user",
            payload_type="message",
            text="remember this after a short interrupt",
        )

        def _brief_interrupt() -> None:
            time.sleep(0.05)
            adapter_module._mark_memory_llm_interrupt()
            _memory_llm_interrupt.set()
            time.sleep(0.02)
            _memory_llm_interrupt.clear()

        interrupt_thread = threading.Thread(target=_brief_interrupt, daemon=True)
        interrupt_thread.start()
        started = time.perf_counter()

        with self.assertRaises(InterruptedError):
            processor.process(envelope)

        interrupt_thread.join(timeout=0.2)
        self.assertLess(time.perf_counter() - started, 0.25)

    def test_memory_processor_warmup_delegates_to_provider(self) -> None:
        router = _FakeTaskRouter()
        processor = MemoryLLMProcessor(task_router=router, keep_alive="30m")

        warmed = processor.warmup()

        self.assertTrue(warmed)
        self.assertEqual(router.provider.warmup_calls, ["30m"])

    def test_worker_pause_resume_and_stop_delegate_to_memory_processor_provider(self) -> None:
        router = _FakeTaskRouter()
        processor = MemoryLLMProcessor(task_router=router)
        worker = BackgroundWorker(
            job_queue=SimpleNamespace(),
            event_store=SimpleNamespace(),
            memory_llm_processor=processor,
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        worker.pause()
        worker.resume()
        worker._running = True
        worker.stop(timeout_sec=0.0)

        self.assertEqual(router.provider.pause_calls, 1)
        self.assertEqual(router.provider.resume_calls, 1)
        self.assertEqual(router.provider.shutdown_calls, 1)

    def test_memory_requests_use_interruptible_streaming_even_for_non_stream_generate(self) -> None:
        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        fake_client = _FakeStreamingClient(
            [
                {"message": {"content": "Hello "}, "done": False},
                {"message": {"content": "world"}, "done": True, "model": "memory-model"},
            ]
        )
        provider._client = fake_client

        payload = provider._chat_once(
            req=LLMRequest(
                model="memory-model",
                messages=[Message(role="user", content="hi")],
                metadata={"source": MemoryLLMProcessor.TASK_NAME, "keep_alive": 0},
            ),
            model="memory-model",
            stream=False,
        )

        self.assertEqual(len(fake_client.chat_calls), 1)
        self.assertTrue(bool(fake_client.chat_calls[0].get("stream")))
        self.assertEqual(str(dict(payload.get("message") or {}).get("content") or ""), "Hello world")

    def test_memory_request_can_be_interrupted_mid_generation(self) -> None:
        def _chunks():
            yield {"message": {"content": "Hello "}, "done": False}
            _memory_llm_interrupt.set()
            yield {"message": {"content": "world"}, "done": True, "model": "memory-model"}

        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        provider._client = _FakeStreamingClient(_chunks)

        with self.assertRaises(InterruptedError):
            provider._chat_once(
                req=LLMRequest(
                    model="memory-model",
                    messages=[Message(role="user", content="hi")],
                    metadata={"source": MemoryLLMProcessor.TASK_NAME, "keep_alive": 0},
                ),
                model="memory-model",
                stream=False,
            )

    def test_memory_service_restarts_worker_when_new_job_arrives(self) -> None:
        service = MemoryService.__new__(MemoryService)
        service.event_store = SimpleNamespace(append=lambda envelope: None)
        service.analyzer = SimpleNamespace(analyze=lambda envelope: {"should_process": True})
        service.job_queue = SimpleNamespace(
            TYPE_MEMORY_LLM_PROCESS="memory",
            enqueue=lambda **kwargs: "job-1",
        )
        service.worker = _FakeWorker(running=False)

        result = service.ingest_event(MemoryEnvelope(text="remember this"))

        self.assertTrue(result["queued"])
        self.assertEqual(service.worker.start_calls, 1)
        self.assertTrue(service.worker.is_running())
        self.assertEqual(service.worker.wake_calls, 0)

    def test_memory_service_does_not_start_worker_while_main_lock_is_held(self) -> None:
        service = MemoryService.__new__(MemoryService)
        service.event_store = SimpleNamespace(append=lambda envelope: None)
        service.analyzer = SimpleNamespace(analyze=lambda envelope: {"should_process": True})
        service.job_queue = SimpleNamespace(
            TYPE_MEMORY_LLM_PROCESS="memory",
            enqueue=lambda **kwargs: "job-1",
        )
        service.worker = _FakeWorker(running=False)

        acquired = _memory_llm_lock.acquire(timeout=0.1)
        try:
            with mock.patch(
                "memory_core.config_manager.get_memory_core_config",
                return_value=SimpleNamespace(memory_llm_scheduler_mode="strict"),
            ):
                result = service.ingest_event(MemoryEnvelope(text="remember this under lock"))
        finally:
            if acquired and _memory_llm_lock.locked():
                _memory_llm_lock.release()

        self.assertTrue(result["queued"])
        self.assertEqual(service.worker.start_calls, 0)

    def test_memory_service_wakes_running_worker_when_job_arrives(self) -> None:
        service = MemoryService.__new__(MemoryService)
        service.event_store = SimpleNamespace(append=lambda envelope: None)
        service.analyzer = SimpleNamespace(analyze=lambda envelope: {"should_process": True})
        service.job_queue = SimpleNamespace(
            TYPE_MEMORY_LLM_PROCESS="memory",
            enqueue=lambda **kwargs: "job-1",
        )
        service.worker = _FakeWorker(running=True)

        result = service.ingest_event(MemoryEnvelope(text="wake worker"))

        self.assertTrue(result["queued"])
        self.assertEqual(service.worker.start_calls, 0)
        self.assertEqual(service.worker.wake_calls, 1)

    def test_memory_service_starts_worker_under_lock_in_cooperative_mode(self) -> None:
        service = MemoryService.__new__(MemoryService)
        service.event_store = SimpleNamespace(append=lambda envelope: None)
        service.analyzer = SimpleNamespace(analyze=lambda envelope: {"should_process": True})
        service.job_queue = SimpleNamespace(
            TYPE_MEMORY_LLM_PROCESS="memory",
            enqueue=lambda **kwargs: "job-1",
        )
        service.worker = _FakeWorker(running=False)

        acquired = _memory_llm_lock.acquire(timeout=0.1)
        try:
            with mock.patch(
                "memory_core.config_manager.get_memory_core_config",
                return_value=SimpleNamespace(memory_llm_scheduler_mode="cooperative"),
            ):
                result = service.ingest_event(MemoryEnvelope(text="remember this under lock"))
        finally:
            if acquired and _memory_llm_lock.locked():
                _memory_llm_lock.release()

        self.assertTrue(result["queued"])
        self.assertEqual(service.worker.start_calls, 1)

    def test_adapter_resume_starts_worker_if_it_was_stopped(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        adapter._enable_pause = True
        adapter._cancel_auto_resume_timer = lambda: None
        adapter._unload_main_model_from_vram = lambda: None
        adapter._resume_memory_llm = lambda: None
        adapter._release_memory_llm_lock = lambda: None
        adapter.service = SimpleNamespace(worker=_FakeWorker(running=False))

        adapter.resume_worker()

        self.assertEqual(adapter.service.worker.start_calls, 1)
        self.assertEqual(adapter.service.worker.resume_calls, 0)
        self.assertEqual(adapter.service.worker.wake_calls, 1)

    def test_adapter_defaults_to_strict_scheduler_mode_when_uninitialized(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        self.assertEqual(adapter.scheduler_mode(), "strict")
        self.assertTrue(adapter.is_strict_scheduler_mode())

    def test_adapter_should_pause_worker_only_in_strict_mode(self) -> None:
        strict_adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        strict_adapter._enable_pause = True
        strict_adapter._scheduler_mode = "strict"
        self.assertTrue(strict_adapter.should_pause_worker_for_api_request())

        cooperative_adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        cooperative_adapter._enable_pause = True
        cooperative_adapter._scheduler_mode = "cooperative"
        self.assertFalse(cooperative_adapter.should_pause_worker_for_api_request())

    def test_pause_worker_does_not_schedule_auto_resume_timer(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        adapter._enable_pause = True
        adapter._pause_timeout = 2.0
        adapter._auto_resume_timer = None
        adapter._cancel_auto_resume_timer = lambda: None
        adapter._acquire_memory_llm_lock = lambda: None
        adapter._pause_memory_llm = lambda: None
        adapter.service = SimpleNamespace(worker=SimpleNamespace(pause=lambda: None))

        adapter.pause_worker()

        self.assertIsNone(adapter._auto_resume_timer)

    def test_adapter_close_shuts_down_memory_provider_even_if_worker_is_already_stopped(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        calls: list[str] = []
        adapter._cancel_auto_resume_timer = lambda: calls.append("cancel_timer")
        adapter._bump_resume_epoch = lambda: calls.append("bump_epoch") or 1
        adapter._release_memory_llm_lock = lambda: calls.append("release_lock")

        worker = SimpleNamespace(
            is_running=lambda: False,
            pause=lambda: calls.append("pause"),
            _shutdown_memory_llm_provider=lambda: calls.append("shutdown_provider"),
        )
        db = SimpleNamespace(close=lambda: calls.append("db_close"))
        adapter.service = SimpleNamespace(
            worker=worker,
            event_store=SimpleNamespace(db=db),
        )

        adapter.close()

        self.assertEqual(
            calls,
            ["cancel_timer", "bump_epoch", "pause", "shutdown_provider", "release_lock", "db_close"],
        )

    def test_worker_marks_itself_stopped_when_thread_is_dead(self) -> None:
        worker = BackgroundWorker(
            job_queue=SimpleNamespace(),
            event_store=SimpleNamespace(),
            memory_llm_processor=SimpleNamespace(),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )
        worker._running = True
        worker._thread = _FakeThread(alive=False)

        self.assertFalse(worker.is_running())
        self.assertIsNone(worker._thread)

    def test_pause_unloads_model_without_closing_client(self) -> None:
        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        provider._unload_known_model = lambda model_name=None: True
        fake_client = _FakeClosableClient()
        old_global_client = getattr(ollama_provider_module.ollama, "_client", None)
        ollama_provider_module.ollama._client = fake_client
        try:
            provider.pause()
        finally:
            ollama_provider_module.ollama._client = old_global_client

        self.assertEqual(fake_client.close_calls, 0)

    def test_resume_recreates_ollama_client(self) -> None:
        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        recreated_client = object()

        with mock.patch.object(ollama_provider_module.ollama, "Client", return_value=recreated_client):
            provider.resume()

        self.assertIs(provider._client, recreated_client)

    def test_warmup_uses_keep_alive_without_generating_tokens(self) -> None:
        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        fake_client = _FakeStreamingClient([])
        provider._client = fake_client

        warmed = provider.warmup(keep_alive="30m")

        self.assertTrue(warmed)
        self.assertEqual(len(fake_client.generate_calls), 1)
        self.assertEqual(fake_client.generate_calls[0].get("keep_alive"), "30m")
        self.assertEqual(dict(fake_client.generate_calls[0].get("options") or {}).get("num_predict"), 0)

    def test_shutdown_uses_zero_keep_alive_for_module_level_unload_attempts(self) -> None:
        provider = OllamaProvider(default_model="memory-model", timeout_sec=1.0)
        provider._current_model = "memory-model"
        provider._unload_known_model = lambda model_name=None: True
        provider_client = _FakeClosableClient()
        global_client = _FakeClosableClient()
        old_global_client = getattr(ollama_provider_module.ollama, "_client", None)
        old_global_module_client = getattr(ollama_provider_module.ollama, "client", None)
        provider._client = provider_client
        ollama_provider_module.ollama._client = global_client
        ollama_provider_module.ollama.client = _FakeClosableClient()
        try:
            with mock.patch.object(ollama_provider_module.ollama, "generate", return_value={}) as generate:
                with mock.patch.object(ollama_provider_module.ollama, "chat", return_value={}) as chat:
                    provider.shutdown()
        finally:
            ollama_provider_module.ollama._client = old_global_client
            ollama_provider_module.ollama.client = old_global_module_client

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(generate.call_args.kwargs.get("keep_alive"), 0)
        self.assertEqual(chat.call_args.kwargs.get("keep_alive"), 0)
        self.assertEqual(provider_client.close_calls, 1)
        self.assertEqual(provider_client.transport_close_calls, 1)
        self.assertEqual(global_client.close_calls, 1)
        self.assertEqual(global_client.transport_close_calls, 1)
        self.assertIsNone(provider._client)

    def test_adapter_resume_prewarms_memory_when_jobs_are_waiting(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        adapter._enable_pause = True
        adapter._cancel_auto_resume_timer = lambda: None
        adapter._unload_main_model_from_vram = lambda: None
        adapter._resume_memory_llm = lambda: None
        adapter._resume_epoch = 0
        adapter._resume_epoch_lock = threading.Lock()
        release_calls: list[str] = []
        adapter._release_memory_llm_lock = lambda: release_calls.append("release")

        worker = _FakeWorker(running=False)
        warmup_calls: list[str] = []
        worker.memory_llm_processor = SimpleNamespace(warmup=lambda: warmup_calls.append("warm") or True)
        worker.job_queue = SimpleNamespace(get_stats=lambda: {"by_type": {"memory_llm_process": 2}})
        adapter.service = SimpleNamespace(worker=worker)

        def _thread_factory(*args, **kwargs):
            target = kwargs["target"]
            return SimpleNamespace(start=lambda: target())

        with mock.patch.object(adapter_module.threading, "Thread", side_effect=_thread_factory):
            adapter.resume_worker()

        self.assertEqual(worker.start_calls, 1)
        self.assertEqual(worker.resume_calls, 0)
        self.assertEqual(warmup_calls, ["warm"])
        self.assertEqual(release_calls, ["release"])
        self.assertEqual(worker.wake_calls, 1)

    def test_adapter_resume_keeps_main_model_loaded_without_pending_memory_jobs(self) -> None:
        adapter = MemoryCoreAdapter.__new__(MemoryCoreAdapter)
        adapter._enable_pause = True
        adapter._cancel_auto_resume_timer = lambda: None
        unload_calls: list[str] = []
        adapter._unload_main_model_from_vram = lambda: unload_calls.append("unload")
        adapter._resume_memory_llm = lambda: None
        adapter._resume_epoch = 0
        adapter._resume_epoch_lock = threading.Lock()
        adapter._release_memory_llm_lock = lambda: None

        worker = _FakeWorker(running=False)
        worker.job_queue = SimpleNamespace(get_stats=lambda: {"by_type": {"memory_llm_process": 0}})
        adapter.service = SimpleNamespace(worker=worker)

        adapter.resume_worker()

        self.assertEqual(unload_calls, [])
        self.assertEqual(worker.start_calls, 1)
        self.assertEqual(worker.wake_calls, 1)

    def test_priority_manager_memory_turn_is_immediate_without_main(self) -> None:
        manager = LLMPriorityManager(enabled=True, wait_timeout=0.1)
        result: dict[str, object] = {}

        thread = threading.Thread(
            target=lambda: result.setdefault("ok", manager.wait_for_turn(LLMPriorityManager.PRIORITY_MEMORY, timeout=0.1)),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=0.2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(bool(result.get("ok")))
        manager.release(LLMPriorityManager.PRIORITY_MEMORY)

    def test_ollama_stream_uses_priority_manager_for_main_requests(self) -> None:
        provider = OllamaProvider(default_model="main-model", timeout_sec=1.0)
        provider._chat_with_retry = lambda **_kwargs: iter(
            [
                {
                    "message": {"content": "hi"},
                    "done": True,
                    "model": "main-model",
                }
            ]
        )
        calls: list[tuple[str, int]] = []

        class _FakePriorityManager:
            def wait_for_turn(self, priority: int, timeout: float | None = None) -> bool:
                calls.append(("wait", priority))
                return True

            def release(self, priority: int) -> None:
                calls.append(("release", priority))

        with mock.patch.object(ollama_provider_module, "get_priority_manager", return_value=_FakePriorityManager()):
            chunks = list(
                provider.stream(
                    LLMRequest(
                        model="main-model",
                        messages=[Message(role="user", content="hello")],
                        metadata={"source": "api"},
                    )
                )
            )

        self.assertEqual([row[0] for row in calls], ["wait", "release"])
        self.assertEqual(calls[0][1], LLMPriorityManager.PRIORITY_MAIN)
        self.assertEqual(calls[1][1], LLMPriorityManager.PRIORITY_MAIN)
        self.assertEqual(len(chunks), 1)

    def test_ollama_generate_retries_without_thinking_when_first_pass_has_no_answer(self) -> None:
        provider = OllamaProvider(default_model="main-model", timeout_sec=1.0)
        payloads = iter(
            [
                {
                    "message": {
                        "content": "",
                        "thinking": "Long chain of thought",
                    },
                    "done": True,
                    "model": "main-model",
                },
                {
                    "message": {
                        "content": "Final answer",
                    },
                    "done": True,
                    "model": "main-model",
                },
            ]
        )
        think_flags: list[bool] = []

        def _chat_with_retry(*, req, model, stream):
            think_flags.append(bool(dict(req.metadata or {}).get("think")))
            self.assertFalse(stream)
            return next(payloads)

        provider._chat_with_retry = _chat_with_retry
        priority_mgr = _FakePriorityManager()

        with mock.patch.object(ollama_provider_module, "get_priority_manager", return_value=priority_mgr):
            response = provider.generate(
                LLMRequest(
                    model="main-model",
                    messages=[Message(role="user", content="hello")],
                    metadata={"source": "api", "think": True},
                )
            )

        self.assertEqual(think_flags, [True, False])
        self.assertEqual(response.text, "Final answer")
        self.assertEqual(response.thinking, "Long chain of thought")
        self.assertEqual(priority_mgr.calls, [("wait", LLMPriorityManager.PRIORITY_MAIN), ("release", LLMPriorityManager.PRIORITY_MAIN)])

    def test_ollama_stream_retries_without_thinking_when_first_pass_has_no_answer(self) -> None:
        provider = OllamaProvider(default_model="main-model", timeout_sec=1.0)
        streams = iter(
            [
                [
                    {
                        "message": {
                            "thinking": "Long ",
                        },
                        "done": False,
                        "model": "main-model",
                    },
                    {
                        "message": {
                            "thinking": "chain of thought",
                        },
                        "done": True,
                        "model": "main-model",
                    },
                ],
                [
                    {
                        "message": {
                            "content": "Final ",
                        },
                        "done": False,
                        "model": "main-model",
                    },
                    {
                        "message": {
                            "content": "answer",
                        },
                        "done": True,
                        "model": "main-model",
                    },
                ],
            ]
        )
        think_flags: list[bool] = []

        def _chat_with_retry(*, req, model, stream):
            think_flags.append(bool(dict(req.metadata or {}).get("think")))
            self.assertTrue(stream)
            return iter(next(streams))

        provider._chat_with_retry = _chat_with_retry
        priority_mgr = _FakePriorityManager()

        with mock.patch.object(ollama_provider_module, "get_priority_manager", return_value=priority_mgr):
            chunks = list(
                provider.stream(
                    LLMRequest(
                        model="main-model",
                        messages=[Message(role="user", content="hello")],
                        metadata={"source": "api", "think": True},
                    )
                )
            )

        self.assertEqual(think_flags, [True, False])
        self.assertEqual("".join(chunk.thinking_delta for chunk in chunks), "Long chain of thought")
        self.assertEqual("".join(chunk.text_delta for chunk in chunks), "Final answer")
        self.assertEqual(sum(1 for chunk in chunks if chunk.done), 1)
        self.assertTrue(chunks[-1].done)
        self.assertEqual(priority_mgr.calls, [("wait", LLMPriorityManager.PRIORITY_MAIN), ("release", LLMPriorityManager.PRIORITY_MAIN)])

    def test_ollama_build_options_preserves_unbounded_num_predict_sentinel(self) -> None:
        options = OllamaProvider._build_options(
            LLMRequest(
                model="main-model",
                messages=[Message(role="user", content="hello")],
                max_tokens=-1,
                metadata={"think": True},
            )
        )

        self.assertEqual(options.get("num_predict"), -1)

    def test_priority_manager_memory_turn_resumes_after_main_release(self) -> None:
        manager = LLMPriorityManager(enabled=True, wait_timeout=0.5)
        self.assertTrue(manager.acquire(LLMPriorityManager.PRIORITY_MAIN))

        result: dict[str, object] = {}

        thread = threading.Thread(
            target=lambda: result.setdefault("ok", manager.wait_for_turn(LLMPriorityManager.PRIORITY_MEMORY, timeout=0.5)),
            daemon=True,
        )
        thread.start()
        time.sleep(0.05)
        manager.release(LLMPriorityManager.PRIORITY_MAIN)
        thread.join(timeout=0.3)

        self.assertFalse(thread.is_alive())
        self.assertTrue(bool(result.get("ok")))
        manager.release(LLMPriorityManager.PRIORITY_MEMORY)

    def test_interrupted_job_requeues_without_backoff(self) -> None:
        job = IngestJob(
            job_id="job-1",
            event_id="event-1",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-1",
            source_kind="user",
            payload_type="message",
            text="hello",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=SimpleNamespace(process=lambda envelope: (_ for _ in ()).throw(InterruptedError("main llm"))),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [("job-1", "")])

    def test_worker_requeues_job_when_memory_processor_detects_interrupt_before_llm_call(self) -> None:
        job = IngestJob(
            job_id="job-2",
            event_id="event-2",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-2",
            source_kind="user",
            payload_type="message",
            text="hello again",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=MemoryLLMProcessor(task_router=_FakeTaskRouter()),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        _memory_llm_interrupt.set()
        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [("job-2", "")])
        self.assertEqual(queue.complete_calls, [])

    def test_worker_unloads_memory_provider_when_queue_becomes_empty(self) -> None:
        worker = BackgroundWorker(
            job_queue=SimpleNamespace(),
            event_store=SimpleNamespace(),
            memory_llm_processor=SimpleNamespace(),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(poll_interval_sec=0.01),
        )
        calls: list[str] = []
        worker.process_one_job = lambda: False
        worker._pause_memory_llm_provider = lambda: calls.append("pause") or setattr(worker, "_memory_llm_provider_unloaded", True)
        worker._wait_or_wake = lambda timeout: worker._stop_event.set()

        worker._run_loop()

        self.assertEqual(calls, ["pause"])
        self.assertTrue(worker._memory_llm_provider_unloaded)

    def test_worker_resumes_memory_provider_before_processing_next_job_after_idle_unload(self) -> None:
        job = IngestJob(
            job_id="job-idle",
            event_id="event-idle",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-idle",
            source_kind="user",
            payload_type="message",
            text="resume after idle unload",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=SimpleNamespace(process=lambda envelope: SimpleNamespace(should_process=False, proposals=[])),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )
        calls: list[str] = []
        worker._memory_llm_provider_unloaded = True
        worker._resume_memory_llm_provider = lambda: calls.append("resume") or setattr(worker, "_memory_llm_provider_unloaded", False)

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(calls, ["resume"])
        self.assertEqual(queue.complete_calls, ["job-idle"])

    def test_worker_requeues_job_when_interrupt_happens_after_memory_result_before_empty_complete(self) -> None:
        job = IngestJob(
            job_id="job-2b",
            event_id="event-2b",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-2b",
            source_kind="user",
            payload_type="message",
            text="hello after result",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )

        def _process(_envelope):
            adapter_module._mark_memory_llm_interrupt()
            return SimpleNamespace(should_process=False, proposals=[])

        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=SimpleNamespace(process=_process),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [("job-2b", "")])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(queue.fail_calls, [])

    def test_worker_retries_job_when_memory_processor_times_out_instead_of_completing(self) -> None:
        job = IngestJob(
            job_id="job-3",
            event_id="event-3",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-3",
            source_kind="user",
            payload_type="message",
            text="slow memory job",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=MemoryLLMProcessor(task_router=_SlowTaskRouter(delay_sec=0.2), timeout_sec=0.05),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(len(queue.fail_calls), 1)
        self.assertIn("hard-timeout", queue.fail_calls[0][1])

    def test_worker_retries_job_when_memory_processor_returns_invalid_json(self) -> None:
        job = IngestJob(
            job_id="job-4",
            event_id="event-4",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-4",
            source_kind="user",
            payload_type="message",
            text="bad json memory job",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=MemoryLLMProcessor(task_router=_InvalidJsonTaskRouter()),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(len(queue.fail_calls), 1)
        self.assertIn("Failed to parse Memory LLM response", queue.fail_calls[0][1])

    def test_worker_requeues_job_when_provider_error_happens_under_interrupt(self) -> None:
        job = IngestJob(
            job_id="job-5",
            event_id="event-5",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-5",
            source_kind="user",
            payload_type="message",
            text="interrupt while provider fails",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )
        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=MemoryLLMProcessor(task_router=_ErrorTaskRouter("connection closed", set_interrupt=True)),
            governor=lambda proposals, envelope: SimpleNamespace(decisions=[], artifacts=[]),
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [("job-5", "")])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(queue.fail_calls, [])

    def test_worker_requeues_job_when_interrupt_happens_after_governor_before_complete(self) -> None:
        job = IngestJob(
            job_id="job-5b",
            event_id="event-5b",
            job_type="memory_llm_process",
            status="queued",
            payload_json="{}",
        )
        event = SimpleNamespace(
            event_id="event-5b",
            source_kind="user",
            payload_type="message",
            text="interrupt after governor",
            metadata={},
            namespace="default",
            workspace_id="global",
            session_id="default",
            ts=0.0,
        )

        def _governor(_proposals, _envelope):
            adapter_module._mark_memory_llm_interrupt()
            return SimpleNamespace(decisions=[], artifacts=[])

        queue = _ImmediateRequeueQueue(job)
        worker = BackgroundWorker(
            job_queue=queue,
            event_store=_EventStoreForInterrupt(event),
            memory_llm_processor=SimpleNamespace(
                process=lambda envelope: SimpleNamespace(
                    should_process=True,
                    proposals=[SimpleNamespace(text="fact")],
                )
            ),
            governor=_governor,
            config=WorkerConfig(),
        )

        handled = worker.process_one_job()

        self.assertTrue(handled)
        self.assertEqual(queue.requeue_calls, [("job-5b", "")])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(queue.fail_calls, [])

    def test_job_queue_clears_stale_error_text_on_dequeue_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = Database(str(Path(tmp_dir) / "queue.db"))
            try:
                queue = JobQueueStore(db)
                job_id = queue.enqueue(
                    event_id="event-cleanup",
                    job_type=JobQueueStore.TYPE_MEMORY_LLM_PROCESS,
                    payload={},
                )

                queue.requeue_immediately(job_id, "transient error")
                row = db.fetchone("SELECT status, error_text FROM ingest_jobs WHERE job_id = ?", (job_id,))
                self.assertEqual(row["status"], "queued")
                self.assertEqual(row["error_text"], "transient error")

                job = queue.dequeue(worker_id="worker-test", job_type=JobQueueStore.TYPE_MEMORY_LLM_PROCESS)
                self.assertIsNotNone(job)
                row = db.fetchone("SELECT status, error_text, locked_by FROM ingest_jobs WHERE job_id = ?", (job_id,))
                self.assertEqual(row["status"], "processing")
                self.assertIsNone(row["error_text"])
                self.assertEqual(row["locked_by"], "worker-test")

                queue.complete(job_id)
                row = db.fetchone("SELECT status, error_text, locked_by, locked_at FROM ingest_jobs WHERE job_id = ?", (job_id,))
                self.assertEqual(row["status"], "done")
                self.assertIsNone(row["error_text"])
                self.assertIsNone(row["locked_by"])
                self.assertIsNone(row["locked_at"])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
