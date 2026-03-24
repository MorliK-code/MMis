"""
Panel Snapshot Builder — адаптер для memory_inspector_panel.py.

Преобразует persisted trace row в формат, который понимает UI панель.
"""

from __future__ import annotations

from typing import Any


def build_panel_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    """
    Строит snapshot для memory_inspector_panel.py.

    Args:
        row: Trace row из trace_store.

    Returns:
        Snapshot в формате для UI панели.
    """
    pipeline = dict(row.get("pipeline") or {})
    memory = dict(row.get("memory") or {})
    links = dict(row.get("links") or {})
    
    out: dict[str, Any] = {
        "request_id": row.get("request_id"),
        "trace_id": row.get("trace_id"),
        "conversation_id": row.get("conversation_id"),
        "turn_id": row.get("turn_id"),
        "created_at": row.get("created_at"),
        "route": row.get("route"),
        "user_text": row.get("user_text"),
        
        # Memory retrieval
        "memory_retrieval": pipeline.get("memory_retrieval", {}),
        
        # Memory governor
        "memory_governor": pipeline.get("memory_governor", {}),
        
        # Active profile
        "active_profile": pipeline.get("active_profile", {}),
        
        # Identity core
        "identity_core": pipeline.get("identity_core", {}),
        
        # Persona snapshot
        "persona_snapshot": pipeline.get("persona_snapshot", {}),
        
        # Active task
        "active_task": pipeline.get("active_task", {}),
        
        # Prompt pack
        "prompt_pack": pipeline.get("prompt_pack", {}),
        
        # Final answer meta
        "final_answer_meta": pipeline.get("final_answer_meta", {}),
        
        # Turn log summaries
        "turn_log_summaries": pipeline.get("turn_log_summaries", {}),
        
        # Turn log warnings
        "turn_log_warnings": list(pipeline.get("turn_log_warnings", [])),
        
        # Memory data
        "event_ids": list(memory.get("event_ids", [])),
        "job_ids": list(memory.get("job_ids", [])),
        "artifact_ids_created": list(memory.get("artifact_ids_created", [])),
        "artifact_ids_updated": list(memory.get("artifact_ids_updated", [])),
        "artifact_ids_superseded": list(memory.get("artifact_ids_superseded", [])),
        "indexed_ids": list(memory.get("indexed_ids", [])),
        
        # Links
        "web_trace_detail_file_path": links.get("web_trace_detail_file_path", ""),
        "web_trace_compact_file_path": links.get("web_trace_compact_file_path", ""),
    }
    
    # Удаляем пустые поля
    return {k: v for k, v in out.items() if v is not None and v != "" and v != []}


def build_worker_panel_snapshot(worker_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Строит snapshot для worker trace.

    Args:
        worker_trace: Список worker trace rows.

    Returns:
        Snapshot в формате для UI панели.
    """
    stages = []
    proposals = []
    decisions = []
    created_ids = []
    indexed_ids = []
    
    for row in worker_trace:
        stage = row.get("stage", "")
        stages.append({
            "stage": stage,
            "created_at": row.get("created_at"),
            "job_id": row.get("job_id"),
            "event_id": row.get("event_id"),
        })
        
        # Собираем proposals
        if stage == "memory_llm_done":
            proposals.extend(row.get("proposals", []))
        
        # Собираем decisions
        if stage == "governor_done":
            decisions.extend(row.get("decisions", []))
            created_ids.extend(row.get("created_ids", []))
        
        # Собираем indexed_ids
        if stage == "vector_index_done":
            indexed_ids.extend(row.get("indexed_ids", []))
    
    return {
        "stages": stages,
        "proposal_count": sum(row.get("proposal_count", 0) for row in worker_trace),
        "proposals": proposals,
        "decisions": decisions,
        "created_ids": created_ids,
        "indexed_ids": indexed_ids,
    }
