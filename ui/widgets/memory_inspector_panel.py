"""Memory Inspector side panel for desktop UI."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _pretty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _short_text(value: Any, limit: int = 72) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _get_section(snapshot: dict[str, Any], key: str, fallback_key: str = "") -> dict[str, Any]:
    value = snapshot.get(key)
    if not value and fallback_key:
        value = snapshot.get(fallback_key)
    return dict(value or {})


def _score_text(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return ""


def _value_hint(value: Any, *, show_raw_scores: bool = True) -> str:
    if isinstance(value, dict):
        if show_raw_scores and "score" in value:
            score = _score_text(value.get("score"))
            if score:
                return f"score={score}"
        if show_raw_scores and "confidence" in value:
            score = _score_text(value.get("confidence"))
            if score:
                return f"confidence={score}"
        if "action" in value:
            return f"action={_short_text(value.get('action'))}"
        if "status" in value:
            return f"status={_short_text(value.get('status'))}"
        return f"dict[{len(value)}]"
    if isinstance(value, list):
        return f"items={len(value)}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return _short_text(value)


def _item_label(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        predicate = str(value.get("predicate") or "").strip()
        raw_value = value.get("value")
        if raw_value is None:
            raw_value = value.get("obj")
        if raw_value is None:
            raw_value = value.get("winner")
        if raw_value is None:
            raw_value = value.get("topic")
        if raw_value is None:
            raw_value = value.get("summary_short")
        if raw_value is None:
            raw_value = value.get("reason")
        if raw_value is None:
            raw_value = value.get("id")
        if predicate and raw_value not in {None, ""}:
            return f"{predicate} = {_short_text(raw_value, 44)}"
        if value.get("group") and value.get("action"):
            return f"{_short_text(value.get('group'), 34)} -> {_short_text(value.get('action'), 20)}"
        if value.get("episode_id") or value.get("source_episode_id"):
            topic = str(value.get("topic") or value.get("summary_short") or value.get("episode_id") or value.get("source_episode_id") or fallback).strip()
            return _short_text(topic, 54)
        if value.get("task_id") or value.get("topic"):
            topic = str(value.get("topic") or value.get("task_id") or fallback).strip()
            return _short_text(topic, 54)
        if value.get("key"):
            return _short_text(value.get("key"), 44)
        if raw_value not in {None, ""}:
            return _short_text(raw_value, 54)
    if isinstance(value, str) and value.strip():
        return _short_text(value, 54)
    return fallback


def _render_details(value: Any, *, indent: int = 0, show_raw_scores: bool = True) -> str:
    pad = "  " * max(0, indent)
    if _is_scalar(value):
        return f"{pad}{value}"
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested in value.items():
            if not show_raw_scores and str(key or "").strip().lower() in {"score", "confidence"}:
                continue
            if _is_scalar(nested):
                lines.append(f"{pad}{key}: {nested}")
                continue
            lines.append(f"{pad}{key}: {_value_hint(nested, show_raw_scores=show_raw_scores)}")
            nested_text = _render_details(nested, indent=indent + 1, show_raw_scores=show_raw_scores).strip()
            if nested_text:
                lines.append(nested_text)
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for index, nested in enumerate(value, start=1):
            label = _item_label(nested, f"item_{index}")
            if _is_scalar(nested):
                lines.append(f"{pad}{index}. {nested}")
                continue
            lines.append(f"{pad}{index}. {label}")
            nested_text = _render_details(nested, indent=indent + 1, show_raw_scores=show_raw_scores).strip()
            if nested_text:
                lines.append(nested_text)
        return "\n".join(lines)
    return f"{pad}{value}"


def _add_child_items(parent: QTreeWidgetItem, value: Any, *, show_raw_scores: bool = True) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not show_raw_scores and str(key or "").strip().lower() in {"score", "confidence"}:
                continue
            child = QTreeWidgetItem([str(key), _value_hint(nested, show_raw_scores=show_raw_scores)])
            child.setData(0, Qt.UserRole, nested)
            parent.addChild(child)
            _add_child_items(child, nested, show_raw_scores=show_raw_scores)
        return
    if isinstance(value, list):
        for index, nested in enumerate(value, start=1):
            label = _item_label(nested, f"item_{index}")
            child = QTreeWidgetItem([label, _value_hint(nested, show_raw_scores=show_raw_scores)])
            child.setData(0, Qt.UserRole, nested)
            parent.addChild(child)
            _add_child_items(child, nested, show_raw_scores=show_raw_scores)


class _InspectorPage(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._show_raw_scores = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(self.splitter, 1)

        self.tree = QTreeWidget(self.splitter)
        self.tree.setObjectName(f"memory_inspector_tree_{title.lower().replace(' ', '_')}")
        self.tree.setHeaderLabels(["Section", "Value"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        right = QWidget(self.splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.mode_tabs = QTabWidget(right)
        self.mode_tabs.setObjectName(f"memory_inspector_modes_{title.lower().replace(' ', '_')}")
        right_layout.addWidget(self.mode_tabs, 1)

        self.details_view = self._make_view("details")
        self.raw_view = self._make_view("raw")
        self.mode_tabs.addTab(self.details_view, "Details")
        self.mode_tabs.addTab(self.raw_view, "Raw JSON")

        self.splitter.setSizes([240, 420])
        self._set_empty()

    @staticmethod
    def _make_view(kind: str) -> QPlainTextEdit:
        view = QPlainTextEdit()
        view.setObjectName(f"memory_inspector_{kind}_view")
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        return view

    def set_show_raw_scores(self, value: bool) -> None:
        self._show_raw_scores = bool(value)

    def set_sections(self, sections: list[tuple[str, Any]]) -> None:
        self.tree.clear()
        for title, value in sections:
            if value in ({}, [], "", None):
                continue
            top = QTreeWidgetItem([str(title), _value_hint(value, show_raw_scores=self._show_raw_scores)])
            top.setData(0, Qt.UserRole, value)
            self.tree.addTopLevelItem(top)
            _add_child_items(top, value, show_raw_scores=self._show_raw_scores)
        if self.tree.topLevelItemCount() <= 0:
            self._set_empty()
            return
        self.tree.expandToDepth(1)
        first = self.tree.topLevelItem(0)
        if first is not None:
            self.tree.setCurrentItem(first)
            self._apply_value(first.data(0, Qt.UserRole))

    def _set_empty(self) -> None:
        self.details_view.setPlainText(f"No {self._title.lower()} trace.")
        self.raw_view.setPlainText("")

    def _apply_value(self, value: Any) -> None:
        self.details_view.setPlainText(
            _render_details(value, show_raw_scores=self._show_raw_scores).strip() or f"No {self._title.lower()} trace."
        )
        self.raw_view.setPlainText(_pretty(value))

    def _on_selection_changed(self) -> None:
        current = self.tree.currentItem()
        if current is None:
            self._set_empty()
            return
        self._apply_value(current.data(0, Qt.UserRole))


class MemoryInspectorPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: dict[str, Any] = {}
        self._memory_inspector_enabled = True
        self._show_raw_scores = False
        self._show_filtered_items = False
        self._show_prompt_blocks = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.title_label = QLabel("Memory Inspector")
        self.title_label.setObjectName("memory_inspector_title")
        self.title_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.title_label)

        self.meta_label = QLabel("No trace yet")
        self.meta_label.setObjectName("memory_inspector_meta")
        self.meta_label.setWordWrap(True)
        self.meta_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.meta_label)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("memory_inspector_tabs")
        root.addWidget(self.tabs, 1)

        self.retrieval_page = _InspectorPage("Retrieval", self.tabs)
        self.governor_page = _InspectorPage("Governor", self.tabs)
        self.identity_core_page = _InspectorPage("Identity Core", self.tabs)
        self.persona_page = _InspectorPage("Persona", self.tabs)
        self.active_task_page = _InspectorPage("Active Task", self.tabs)
        self.prompt_page = _InspectorPage("Prompt", self.tabs)
        self.loop_page = _InspectorPage("Loop", self.tabs)

        self.tabs.addTab(self.retrieval_page, "Retrieval")
        self.tabs.addTab(self.governor_page, "Governor")
        self.tabs.addTab(self.identity_core_page, "Identity Core")
        self.tabs.addTab(self.persona_page, "Persona")
        self.tabs.addTab(self.active_task_page, "Active Task")
        self.tabs.addTab(self.prompt_page, "Prompt")
        self.tabs.addTab(self.loop_page, "Loop")

        self.configure_display(
            enabled=self._memory_inspector_enabled,
            show_raw_scores=self._show_raw_scores,
            show_filtered_items=self._show_filtered_items,
            show_prompt_blocks=self._show_prompt_blocks,
        )

    def clear_snapshot(self) -> None:
        self.set_snapshot({})

    def configure_display(
        self,
        *,
        enabled: bool | None = None,
        show_raw_scores: bool | None = None,
        show_filtered_items: bool | None = None,
        show_prompt_blocks: bool | None = None,
    ) -> None:
        if enabled is not None:
            self._memory_inspector_enabled = bool(enabled)
        if show_raw_scores is not None:
            self._show_raw_scores = bool(show_raw_scores)
        if show_filtered_items is not None:
            self._show_filtered_items = bool(show_filtered_items)
        if show_prompt_blocks is not None:
            self._show_prompt_blocks = bool(show_prompt_blocks)
        for page in (
            self.retrieval_page,
            self.governor_page,
            self.identity_core_page,
            self.persona_page,
            self.active_task_page,
            self.prompt_page,
            self.loop_page,
        ):
            page.set_show_raw_scores(self._show_raw_scores)
        self.setVisible(self._memory_inspector_enabled)
        self.set_snapshot(self._snapshot)

    def set_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        row = dict(snapshot or {})
        self._snapshot = row
        self.meta_label.setText(self._build_meta_text(row))
        self.retrieval_page.set_sections(self._retrieval_sections(row))
        self.governor_page.set_sections(self._governor_sections(row))
        self.identity_core_page.set_sections(self._identity_core_sections(row))
        self.persona_page.set_sections(self._persona_sections(row))
        self.active_task_page.set_sections(self._active_task_sections(row))
        self.prompt_page.set_sections(self._prompt_sections(row))
        self.loop_page.set_sections(self._loop_sections(row))

    @staticmethod
    def _build_meta_text(snapshot: dict[str, Any]) -> str:
        request_id = str(snapshot.get("request_id") or "").strip()
        user_text = str(snapshot.get("user_text") or "").strip()
        if not request_id and not user_text:
            return "No trace yet"
        lines: list[str] = []
        if request_id:
            lines.append(f"request_id: {request_id}")
        if user_text:
            lines.append(f"user: {user_text}")
        return "\n".join(lines)

    def _retrieval_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        retrieval = _get_section(snapshot, "retrieval", "memory_retrieval")
        return [
            (
                "Query Plan",
                {
                    "query": retrieval.get("query"),
                    "recall_mode": retrieval.get("recall_mode"),
                    "request_plan": retrieval.get("request_plan"),
                    "fanout_sources": retrieval.get("fanout_sources"),
                    "memory_block_keys": retrieval.get("memory_block_keys"),
                    "selected_total": retrieval.get("selected_total"),
                },
            ),
            ("Fan-out Queries", retrieval.get("fanout_queries")),
            ("Exact Hits", retrieval.get("exact_self_fact_hits")),
            ("Hit Summary", retrieval.get("confidence")),
            ("Selected Facts", retrieval.get("selected_facts")),
            ("Selected Episodes", retrieval.get("selected_episodes")),
            ("Selected Claims", retrieval.get("selected_claims")),
            ("Selected Documents", retrieval.get("selected_documents")),
            ("Selected Messages", retrieval.get("selected_messages")),
            ("Rerank", retrieval.get("score_breakdowns") if self._show_raw_scores else []),
            ("Task Continuity", retrieval.get("task_continuity")),
            ("Filtered Out", retrieval.get("filtered_out") if self._show_filtered_items else []),
            ("Truncated", retrieval.get("truncated")),
        ]

    def _governor_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        governor = _get_section(snapshot, "governor", "memory_governor")
        active_profile = dict(snapshot.get("active_profile") or {})
        return [
            ("Active Profile", active_profile),
            ("Decisions", governor.get("decisions")),
            ("Superseded", governor.get("superseded")),
            ("Conflicts", governor.get("conflicts")),
        ]

    def _persona_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        persona = dict(snapshot.get("persona_snapshot") or {})
        return [
            (
                "Overview",
                {
                    "mood": persona.get("mood"),
                    "active_mode": persona.get("active_mode"),
                },
            ),
            ("Active Task Bridge", persona.get("active_task")),
            ("Relation Continuity", persona.get("relation_continuity")),
            ("Recent User State", persona.get("recent_user_state")),
            ("User Addressing", persona.get("user_addressing")),
            ("Stable Traits", persona.get("stable_traits")),
            ("Relation State", persona.get("relation_state")),
            ("Boundaries", persona.get("boundaries")),
            ("Emotional Handling", persona.get("emotional_handling")),
            ("Response Bias", persona.get("response_bias")),
            ("User Profile Hints", persona.get("user_profile_hints")),
            ("Debug", persona.get("debug")),
        ]

    def _identity_core_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        identity_core = dict(snapshot.get("identity_core") or {})
        snapshot_row = dict(identity_core.get("snapshot") or identity_core.get("memory_snapshot") or {})
        sources = dict(
            identity_core.get("sources")
            or dict(dict(snapshot_row.get("debug") or {}).get("sources") or {})
        )
        trait_baselines = dict(
            identity_core.get("trait_baselines")
            or dict(snapshot_row.get("assistant_trait_baseline") or {})
        )
        return [
            (
                "Overview",
                {
                    "character_id": snapshot_row.get("character_id"),
                    "updated_at": snapshot_row.get("updated_at"),
                },
            ),
            ("Snapshot", snapshot_row),
            ("Active Keys", identity_core.get("active_identity_keys")),
            ("Trait Baselines", trait_baselines),
            ("Protected Keys", identity_core.get("protected_keys")),
            ("Pending Overrides", identity_core.get("pending_overrides")),
            ("Recent Decisions", identity_core.get("recent_decisions")),
            ("Sources", sources),
        ]

    def _active_task_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        active_task = dict(snapshot.get("active_task") or {})
        return [
            (
                "Task",
                {
                    "task_id": active_task.get("task_id"),
                    "topic": active_task.get("topic"),
                    "status": active_task.get("status"),
                    "source_episode_id": active_task.get("source_episode_id"),
                    "confidence": active_task.get("confidence"),
                    "updated_at": active_task.get("updated_at"),
                    "source": active_task.get("source"),
                    "reason": active_task.get("reason"),
                    "event": active_task.get("event"),
                },
            ),
            ("Summary", active_task.get("summary_short")),
            ("Current Goal", active_task.get("current_goal")),
            ("Next Steps", active_task.get("next_steps")),
            ("Open Questions", active_task.get("open_questions")),
            ("Decisions", active_task.get("decisions")),
        ]

    def _prompt_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        prompt_pack = dict(snapshot.get("prompt_pack") or {})
        sections = [
            (
                "Pack",
                {
                    "token_estimate": prompt_pack.get("token_estimate"),
                    "section_keys": prompt_pack.get("section_keys"),
                },
            ),
            ("Included Memory Blocks", prompt_pack.get("included_memory_blocks")),
            ("Omitted Memory Blocks", prompt_pack.get("omitted_memory_blocks")),
        ]
        if self._show_prompt_blocks:
            sections.extend(
                [
                    ("Memory Block", prompt_pack.get("memory_block")),
                    ("Persona Block", prompt_pack.get("persona_block")),
                    ("Active Task Block", prompt_pack.get("active_task_block")),
                ]
            )
        return sections

    def _loop_sections(self, snapshot: dict[str, Any]) -> list[tuple[str, Any]]:
        final_meta = dict(snapshot.get("final_answer_meta") or {})
        tool_loop = dict(snapshot.get("tool_loop") or final_meta.get("tool_loop") or {})
        return [
            (
                "Overview",
                {
                    "route": final_meta.get("route"),
                    "served_model": final_meta.get("served_model"),
                    "agent_loop": _get_section({"tool_loop": tool_loop}, "tool_loop").get("agent_loop", final_meta.get("agent_loop")),
                    "agent_tool_calls": _get_section({"tool_loop": tool_loop}, "tool_loop").get("agent_tool_calls", final_meta.get("agent_tool_calls")),
                    "agent_passes": _get_section({"tool_loop": tool_loop}, "tool_loop").get("agent_passes", final_meta.get("agent_passes")),
                    "memory_reasoning_used": final_meta.get("memory_reasoning_used"),
                    "memory_reasoning_sections": final_meta.get("memory_reasoning_sections"),
                    "output_len": final_meta.get("output_len"),
                },
            ),
            ("Iterations", tool_loop.get("iterations")),
            ("Executed Tools", tool_loop.get("executed_tools")),
            (
                "Memory Reasoning",
                tool_loop.get("memory_reasoning_snapshot") or final_meta.get("memory_reasoning_snapshot"),
            ),
            ("Warnings", final_meta.get("warnings")),
            ("Errors", final_meta.get("errors")),
        ]
