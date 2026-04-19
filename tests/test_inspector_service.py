from __future__ import annotations

import unittest
from types import SimpleNamespace

from memory_core.episode_manager import EpisodeManager
from memory_core.inspect.inspector_service import MemoryInspectorService
from memory_core.runtime_session_store import RuntimeSessionStore
from memory_core.schemas import MemoryEnvelope


class _FakeJobQueue:
    def get_stats(self):
        return {
            "queued": 0,
            "processing": 0,
            "retry_wait": 0,
            "done": 11,
            "dead": 0,
            "by_type": {},
        }

    def list_jobs(self, status=None, limit=5000):
        jobs = [
            SimpleNamespace(
                job_id="job-first-run",
                event_id="event-1",
                job_type="memory_llm_process",
                status="done",
                priority=5,
                attempts=0,
                max_attempts=3,
                locked_by=None,
                locked_at=None,
                available_at=0.0,
                error_text=None,
                created_at=1.0,
                updated_at=2.0,
            ),
            SimpleNamespace(
                job_id="job-restarted",
                event_id="event-2",
                job_type="memory_llm_process",
                status="done",
                priority=5,
                attempts=1,
                max_attempts=3,
                locked_by=None,
                locked_at=None,
                available_at=0.0,
                error_text=None,
                created_at=3.0,
                updated_at=4.0,
            ),
        ]
        if status is None:
            return jobs[:limit]
        return [job for job in jobs if job.status == status][:limit]


class _FakeWorker:
    def get_stats(self):
        return {
            "running": True,
            "config": {"scheduler_mode": "cooperative"},
            "stats": {
                "jobs_processed": 15,
                "jobs_succeeded": 11,
                "jobs_failed": 0,
                "jobs_retried": 4,
                "interrupt_count": 3,
                "requeue_count": 3,
            },
        }


class _FakeTraceStore:
    def get_worker_trace(self, trace_id: str):
        if trace_id == "trace_job-first-run":
            return [
                {"stage": "event_loaded"},
                {"stage": "memory_llm_done"},
                {"stage": "governor_done"},
                {"stage": "job_done", "created_at": "2026-03-27T00:00:01"},
            ]
        if trace_id == "trace_job-restarted":
            return [
                {"stage": "event_loaded"},
                {"stage": "event_loaded"},
                {"stage": "memory_llm_done"},
                {"stage": "governor_done"},
                {"stage": "job_done", "created_at": "2026-03-27T00:00:02"},
            ]
        return []


class InspectorServiceTests(unittest.TestCase):
    def test_overview_exposes_worker_retry_counters_separately_from_done_jobs(self) -> None:
        memory_core = SimpleNamespace(
            service=SimpleNamespace(
                job_queue=_FakeJobQueue(),
                worker=_FakeWorker(),
                trace_store=_FakeTraceStore(),
            )
        )

        overview = MemoryInspectorService(memory_core).get_overview()

        self.assertEqual(overview["jobs_done"], 11)
        self.assertEqual(overview["jobs_done_confirmed"], 2)
        self.assertEqual(overview["jobs_done_first_run"], 1)
        self.assertEqual(overview["jobs_done_after_restart"], 1)
        self.assertEqual(overview["jobs_restarted_total"], 1)
        self.assertEqual(overview["worker_jobs_processed"], 15)
        self.assertEqual(overview["worker_jobs_succeeded"], 11)
        self.assertEqual(overview["worker_jobs_retried"], 4)
        self.assertEqual(overview["worker_jobs_failed"], 0)
        self.assertEqual(overview["worker_interrupt_count"], 3)
        self.assertEqual(overview["worker_requeue_count"], 3)
        self.assertEqual(overview["scheduler_mode"], "cooperative")

    def test_list_jobs_exposes_trace_confirmation_and_restart_counts(self) -> None:
        memory_core = SimpleNamespace(
            service=SimpleNamespace(
                job_queue=_FakeJobQueue(),
                trace_store=_FakeTraceStore(),
            )
        )

        jobs = MemoryInspectorService(memory_core).list_jobs(limit=10)
        by_id = {job["id"]: job for job in jobs}

        self.assertEqual(by_id["job-first-run"]["run_count"], 1)
        self.assertEqual(by_id["job-first-run"]["restart_count"], 0)
        self.assertTrue(by_id["job-first-run"]["trace_has_job_done"])
        self.assertEqual(by_id["job-first-run"]["completion_label"], "confirmed_done_first_run")

        self.assertEqual(by_id["job-restarted"]["run_count"], 2)
        self.assertEqual(by_id["job-restarted"]["restart_count"], 1)
        self.assertEqual(by_id["job-restarted"]["inferred_interruptions"], 0)
        self.assertTrue(by_id["job-restarted"]["trace_has_job_done"])
        self.assertEqual(by_id["job-restarted"]["completion_label"], "confirmed_done_after_restart")

    def test_runtime_snapshot_exposes_sessions_and_hidden_episodes(self) -> None:
        runtime_store = RuntimeSessionStore()
        episode_manager = EpisodeManager()
        envelope = MemoryEnvelope(
            text="Runtime state should be visible immediately",
            workspace_id="asya",
            session_id="chat-runtime",
        )
        episode = episode_manager.update_from_event(envelope)
        runtime_store.update_from_event(envelope, current_episode_id=episode["episode_id"])
        memory_core = SimpleNamespace(
            service=SimpleNamespace(
                runtime_session_store=runtime_store,
                episode_manager=episode_manager,
            )
        )

        service = MemoryInspectorService(memory_core)
        overview = service.get_overview()
        runtime = service.get_runtime(limit=10)

        self.assertEqual(overview["runtime_sessions_count"], 1)
        self.assertEqual(overview["runtime_episodes_count"], 1)
        self.assertEqual(runtime["sessions"][0]["session_id"], "chat-runtime")
        self.assertEqual(runtime["active_episode"]["episode_id"], episode["episode_id"])
        self.assertIn("scheduler_mode", runtime)
        self.assertIn("interrupt_count", runtime)
        self.assertIn("requeue_count", runtime)


if __name__ == "__main__":
    unittest.main()
