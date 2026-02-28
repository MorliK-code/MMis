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
]
