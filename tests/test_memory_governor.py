from __future__ import annotations

import time

from memory.governor import FACT_GROUPS, GROUP_POLICY, MemoryGovernor
from memory.memory_models import MemoryLevel, MemoryRecord, MemoryScope, MemoryStatus, MemoryType


def _fact_record(
    record_id: str,
    *,
    predicate: str,
    value: str,
    confidence: float = 0.7,
    canonical_key: str | None = None,
    namespace: str = "default",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    relation: str = "",
    source_role: str = "user",
    source_kind: str = "structured_fact",
) -> MemoryRecord:
    now = time.time()
    return MemoryRecord(
        id=record_id,
        text=f"user.{predicate}={value}",
        memory_type=MemoryType.FACT,
        level=MemoryLevel.L3_SEMANTIC,
        scope=MemoryScope.CONVERSATION,
        namespace=namespace,
        metadata={
            "canonical_key": str(canonical_key or f"user.{predicate}"),
            "fact": {
                "subject": "user",
                "predicate": predicate,
                "value": value,
                "confidence": confidence,
                "source_role": source_role,
                "source_kind": source_kind,
            },
            "relation": relation,
        },
        importance=0.7,
        confidence=confidence,
        created_at=now,
        updated_at=now,
        status=status,
    )


def test_governor_keeps_active_when_no_candidates() -> None:
    governor = MemoryGovernor()
    new_record = _fact_record("fact:new", predicate="environment_gpu_model", value="RTX 3050 Ti", confidence=0.82)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[])

    assert decision.action == "keep_active"
    assert decision.reason == "no_active_conflict"
    assert decision.winner_record_id == "fact:new"


def test_governor_supersedes_lower_confidence_same_group_fact() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record("fact:old", predicate="environment_ram_gb", value="16", confidence=0.61)
    new_record = _fact_record("fact:new", predicate="environment_memory_gb", value="32", confidence=0.87)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "supersede_old"
    assert decision.reason == "singleton_group_higher_score"
    assert decision.winner_record_id == "fact:new"
    assert decision.loser_record_id == "fact:old"


def test_governor_keeps_parallel_on_close_conflict() -> None:
    governor = MemoryGovernor(parallel_margin=0.05)
    old_record = _fact_record("fact:old", predicate="environment_os", value="Windows 11", confidence=0.79)
    new_record = _fact_record("fact:new", predicate="environment_os", value="Linux", confidence=0.82)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "supersede_old"
    assert decision.reason == "singleton_group_higher_score"
    assert decision.loser_record_id == "fact:old"


def test_governor_noop_when_existing_same_value_wins() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record("fact:old", predicate="environment_runtime_python", value="python 3.11", confidence=0.88)
    new_record = _fact_record("fact:new", predicate="environment_runtime_python", value="python 3.11", confidence=0.70)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "noop"
    assert decision.reason == "same_value_existing_kept"
    assert decision.winner_record_id == "fact:old"
    assert decision.loser_record_id == "fact:new"


def test_governor_rebuild_profile_snapshot_marks_conflicts_and_filters_namespace() -> None:
    governor = MemoryGovernor()
    windows_record = _fact_record("fact:win", predicate="environment_os", value="Windows 11", confidence=0.76, namespace="profile-a")
    linux_record = _fact_record("fact:linux", predicate="environment_os", value="Linux", confidence=0.81, namespace="profile-a")
    other_namespace = _fact_record("fact:other", predicate="environment_os", value="macOS", confidence=0.84, namespace="profile-b")

    snapshot = governor.rebuild_profile_snapshot(
        namespace="profile-a",
        active_fact_rows=[windows_record, linux_record, other_namespace],
    )

    assert snapshot.namespace == "profile-a"
    assert len(snapshot.active_facts) == 2
    assert {item["record_id"] for item in snapshot.active_facts.values()} == {"fact:win", "fact:linux"}
    assert snapshot.conflicts == [
        {
            "group": "environment.os",
            "record_ids": ["fact:win", "fact:linux"],
            "values": ["Windows 11", "Linux"],
        }
    ]


def test_governor_keeps_parallel_for_multi_value_project_name_group() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record("fact:old", predicate="project_name", value="MMis", confidence=0.81)
    new_record = _fact_record("fact:new", predicate="project_name", value="AnotherProject", confidence=0.86)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "keep_parallel"
    assert decision.reason == "multi_value_group_allowed"
    assert decision.parallel_with_record_id == "fact:old"


def test_governor_keeps_parallel_for_multi_value_interest_group() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record("fact:old", predicate="interest", value="flowers", confidence=0.72)
    new_record = _fact_record("fact:new", predicate="interest", value="music", confidence=0.75)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "keep_parallel"
    assert decision.reason == "multi_value_group_allowed"
    assert decision.parallel_with_record_id == "fact:old"


def test_governor_keeps_parallel_for_preferred_tool_group() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record("fact:old", predicate="preferred_tool", value="VS Code", confidence=0.73)
    new_record = _fact_record("fact:new", predicate="preferred_tool", value="PyCharm", confidence=0.78)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "keep_parallel"
    assert decision.reason == "multi_value_group_allowed"
    assert decision.parallel_with_record_id == "fact:old"


def test_governor_declares_fact_groups_and_group_policy() -> None:
    assert FACT_GROUPS["environment_runtime_python"] == "environment.python"
    assert FACT_GROUPS["environment_memory_gb"] == "environment.ram"
    assert FACT_GROUPS["preferred_language"] == "preferences.language"
    assert GROUP_POLICY["environment.python"] == "singleton"
    assert GROUP_POLICY["preferences.language"] == "soft_singleton"
    assert GROUP_POLICY["project.name"] == "multi"


def test_governor_soft_singleton_allows_parallel_on_close_scores() -> None:
    governor = MemoryGovernor(parallel_margin=0.04)
    old_record = _fact_record("fact:old", predicate="preferred_language", value="python", confidence=0.80)
    new_record = _fact_record("fact:new", predicate="preferred_language", value="rust", confidence=0.82)

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "keep_parallel"
    assert decision.reason == "soft_singleton_scores_close_parallel"
    assert decision.parallel_with_record_id == "fact:old"


def test_governor_rebuild_profile_snapshot_splits_stable_and_volatile_layers() -> None:
    governor = MemoryGovernor()
    directness = _fact_record(
        "fact:direct",
        predicate="assistant_directness",
        value="0.82",
        confidence=0.86,
        namespace="profile-layers",
    )
    editor = _fact_record(
        "fact:editor",
        predicate="preferred_editor",
        value="vscode",
        confidence=0.80,
        namespace="profile-layers",
    )
    project = _fact_record(
        "fact:project",
        predicate="project_name",
        value="MMis",
        confidence=0.78,
        namespace="profile-layers",
    )

    snapshot = governor.rebuild_profile_snapshot(
        namespace="profile-layers",
        active_fact_rows=[directness, editor, project],
    )

    assert list(snapshot.persistent_traits.keys()) == ["user.assistant_directness"]
    assert list(snapshot.volatile_preferences.keys()) == ["user.preferred_editor"]
    assert list(snapshot.session_preferences.keys()) == ["user.project_name"]
    assert snapshot.resolved_profile["assistant_directness"] == "0.82"
    assert snapshot.resolved_profile["preferred_editor"] == "vscode"
    assert snapshot.resolved_profile["project_name"] == ["MMis"]


def test_governor_blocks_protected_identity_override_without_confirmation() -> None:
    governor = MemoryGovernor()
    old_record = _fact_record(
        "fact:old",
        predicate="identity_name",
        value="Pasha",
        confidence=0.91,
        relation="identity",
    )
    new_record = _fact_record(
        "fact:new",
        predicate="identity_name",
        value="Pashka",
        confidence=0.94,
        relation="identity",
    )

    decision = governor.decide_for_fact(new_record=new_record, active_candidates=[old_record])

    assert decision.action == "noop"
    assert decision.reason == "protected_profile_requires_confirmation"
    assert decision.winner_record_id == "fact:old"
