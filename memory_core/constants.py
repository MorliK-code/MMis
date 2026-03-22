"""
Константы memory_core.
"""

DEFAULT_NAMESPACE = "default"
DEFAULT_WORKSPACE = "global"
DEFAULT_SESSION = "default"
DEFAULT_TOP_K = 8

# Источник событий
SOURCE_USER = "user"
SOURCE_ASSISTANT = "assistant"
SOURCE_TOOL = "tool"
SOURCE_SYSTEM = "system"
SOURCE_DOCUMENT = "document"

# Типы артефактов
ARTIFACT_FACT = "fact"
ARTIFACT_PROFILE_FACT = "profile_fact"
ARTIFACT_EPISODE = "episode"
ARTIFACT_TASK = "task"
ARTIFACT_DOCUMENT_CHUNK = "document_chunk"
ARTIFACT_DOCUMENT_SUMMARY = "document_summary"
ARTIFACT_NOTE = "note"

# Статусы артефактов
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"
STATUS_SUPERSEDED = "superseded"

# Типы связей между артефактами
LINK_DERIVED_FROM = "derived_from"
LINK_DUPLICATES = "duplicates"
LINK_SUPERSEDES = "supersedes"
LINK_BELONGS_TO_WORKSPACE = "belongs_to_workspace"
LINK_SUPPORTS_TASK = "supports_task"
LINK_SAME_SUBJECT = "same_subject"

# Версии схем
SCHEMA_VERSION = "1.0.0"
