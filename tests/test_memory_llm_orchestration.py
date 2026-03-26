from __future__ import annotations

import json
import time
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import memory_core.adapter as adapter_module
import llm.ollama_provider as ollama_provider_module
from llm.ollama_provider import OllamaProvider, _memory_llm_interrupt
from llm.priority_manager import LLMPriorityManager
from llm.provider_base import LLMRequest, Message
from memory_core.adapter import MemoryCoreAdapter, _memory_llm_lock
from memory_core.facade import MemoryService
from memory_core.processors.memory_llm_processor import MemoryLLMProcessor
from memory_core.storage.job_queue_store import IngestJob
from memory_core.schemas import MemoryEnvelope
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

    def close(self) -> None:
        self.close_calls += 1


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
        self.assertEqual(dict(call.get("metadata") or {}).get("source"), MemoryLLMProcessor.TASK_NAME)
        self.assertEqual(dict(call.get("metadata") or {}).get("keep_alive"), "30m")

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
        self.assertNotIn("keep_alive", metadata)

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
        self.assertEqual(queue.requeue_calls, [("job-1", "main llm")])

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
        self.assertEqual(queue.requeue_calls, [("job-2", "Memory LLM interrupted by main model")])
        self.assertEqual(queue.complete_calls, [])

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
        self.assertEqual(queue.requeue_calls, [("job-5", "Memory LLM interrupted by main model")])
        self.assertEqual(queue.complete_calls, [])
        self.assertEqual(queue.fail_calls, [])


if __name__ == "__main__":
    unittest.main()
