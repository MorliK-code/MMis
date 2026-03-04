from __future__ import annotations

from modules.automation import BrowserConfig, BrowserController, OSActions, TaskExecutor
from modules.character import CharacterEngine
from modules.internet import SearchClient, WebScraper
from modules.screen import OCRExtractor, ScreenAnalyzer
from modules.voice import STTService, TTSService, VoiceManager

__all__ = [
    "TTSService",
    "STTService",
    "VoiceManager",
    "OCRExtractor",
    "ScreenAnalyzer",
    "OSActions",
    "BrowserController",
    "BrowserConfig",
    "TaskExecutor",
    "SearchClient",
    "WebScraper",
    "CharacterEngine",
    "CharacterRuntime",
]


# Lazy import for CharacterRuntime to avoid circular dependency.
def __getattr__(name: str):
    if name == "CharacterRuntime":
        from core.character_runtime import CharacterRuntime as _CharacterRuntime
        return _CharacterRuntime
    raise AttributeError(name)
