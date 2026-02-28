from __future__ import annotations

from importlib import import_module

__all__ = [
    "PROFILE_FAST",
    "PROFILE_BALANCED",
    "PROFILE_QUALITY",
    "IntentMeta",
    "EmotionMeta",
    "Metadata",
    "MetadataExtractor",
    "extract",
    "extract_message_metadata",
]

_LAZY = {
    "PROFILE_FAST": "metadata.metadata_extractor",
    "PROFILE_BALANCED": "metadata.metadata_extractor",
    "PROFILE_QUALITY": "metadata.metadata_extractor",
    "IntentMeta": "metadata.metadata_extractor",
    "EmotionMeta": "metadata.metadata_extractor",
    "Metadata": "metadata.metadata_extractor",
    "MetadataExtractor": "metadata.metadata_extractor",
    "extract": "metadata.metadata_extractor",
    "extract_message_metadata": "metadata.metadata_extractor",
}


def __getattr__(name: str):
    module_path = _LAZY.get(name)
    if not module_path:
        raise AttributeError(name)
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
