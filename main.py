from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any

from config.logging_config import setup_logging
from config.model_config import ModelProfile, get_profile
from config.paths import BASE_DIR, ensure_dirs
from config.settings import AppSettings, load_config
from core.brain import Brain
from core.character_runtime import CharacterRuntime
from core.response_pipeline import ResponsePipeline
from core.spec_registry import validate_no_txt_paths
from llm import build_provider
from llm.provider_base import LLMProviderBase
from llm.tokenizer import ApproxTokenizer, Tokenizer
from memory.event_store import EventStore
from memory.fact_extractor import FactExtractor
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.profile_store import AssistantProfileStore, UserProfileStore
from memory.short_memory import ShortMemory
from memory.vector_store import VectorStore
from metadata.metadata_extractor import MetadataExtractor
from modules.automation import BrowserConfig, BrowserController, OSActions, OSActionConfig, TaskExecutor
from modules.internet import SearchClient, WebScraper
from modules.screen import ScreenAnalyzer
from modules.voice import VoiceManager


LOGGER = logging.getLogger(__name__)


@dataclass
class AppContainer:
    settings: AppSettings
    profile: ModelProfile
    provider: LLMProviderBase
    tokenizer: Tokenizer
    character_runtime: CharacterRuntime
    metadata_extractor: MetadataExtractor
    memory_manager: MemoryManager
    response_pipeline: ResponsePipeline
    brain: Brain
    voice: VoiceManager | None = None
    screen: ScreenAnalyzer | None = None
    automation: TaskExecutor | None = None
    internet_search: SearchClient | None = None
    internet_scraper: WebScraper | None = None

    def shutdown(self) -> None:
        _safe_call(self.voice, "shutdown")
        _safe_call(self.character_runtime, "save")
        _safe_call(self.memory_manager.short_memory, "save")
        _safe_call(self.memory_manager.long_memory, "save")
        _safe_call(self.memory_manager.vector_store, "save")
        _safe_call(self.memory_manager.user_profile_store, "save")
        _safe_call(self.memory_manager.assistant_profile_store, "save")
        _safe_call(self.provider, "shutdown")
        _safe_call(self.provider, "close")
        logging.shutdown()


def build_container(settings: AppSettings) -> AppContainer:
    validate_no_txt_paths(settings)
    ensure_dirs(memory_dir=settings.memory_dir)
    profile = get_profile(settings.active_profile)
    provider_name = _resolve_provider_name(settings.llm_default_provider)
    provider = build_provider(provider_name, default_model=settings.model_name)
    tokenizer: Tokenizer = ApproxTokenizer()

    character_runtime = CharacterRuntime()
    character_runtime.set_quality_profile(_state_profile_name(settings.active_profile))
    metadata_extractor = MetadataExtractor(cache_size=280)

    short_memory = ShortMemory(limit=80, summary_trigger=60)
    long_memory = LongMemory()
    vector_store = VectorStore(dim=128)
    event_store = EventStore()
    fact_extractor = FactExtractor()
    user_profile_store = UserProfileStore()
    assistant_profile_store = AssistantProfileStore()
    memory_manager = MemoryManager(
        short_memory=short_memory,
        long_memory=long_memory,
        vector_store=vector_store,
        fact_extractor=fact_extractor,
        user_profile_store=user_profile_store,
        assistant_profile_store=assistant_profile_store,
        event_store=event_store,
        retrieve_score_threshold=0.28,
    )

    response_pipeline = ResponsePipeline(
        provider=provider,
        character_runtime=character_runtime,
        metadata_extractor=metadata_extractor,
        memory_manager=memory_manager,
    )
    brain = Brain(
        provider=provider,
        state_manager=character_runtime,
        memory_manager=memory_manager,
        metadata_extractor=metadata_extractor,
        response_pipeline=response_pipeline,
    )

    voice = VoiceManager() if settings.voice_enabled else None
    screen = ScreenAnalyzer() if settings.screen_enabled else None
    automation = _build_automation(settings=settings, event_store=event_store) if settings.automation_enabled else None
    internet_search = (
        SearchClient(
            endpoint=settings.search_api_url,
            provider=settings.search_provider,
            strict_endpoint=bool(settings.search_strict_endpoint),
            timeout_s=float(settings.search_timeout_sec),
        )
        if settings.internet_enabled
        else None
    )
    internet_scraper = (
        WebScraper(
            timeout_s=int(settings.web_fetch_timeout_sec),
            retries=int(settings.web_fetch_retries),
            clean_max_chars=int(settings.web_clean_max_chars),
            clean_min_chars=int(settings.web_clean_min_chars),
            clean_language_hint=str(settings.web_clean_language_hint or ""),
        )
        if settings.internet_enabled
        else None
    )

    return AppContainer(
        settings=settings,
        profile=profile,
        provider=provider,
        tokenizer=tokenizer,
        character_runtime=character_runtime,
        metadata_extractor=metadata_extractor,
        memory_manager=memory_manager,
        response_pipeline=response_pipeline,
        brain=brain,
        voice=voice,
        screen=screen,
        automation=automation,
        internet_search=internet_search,
        internet_scraper=internet_scraper,
    )


def setup_app() -> AppContainer:
    """Backward-compatible app bootstrap used by legacy debug scripts/tests."""
    settings = load_config()
    ensure_dirs(memory_dir=settings.memory_dir)
    setup_logging(settings)
    return build_container(settings)


def run_cli(container: AppContainer, *, once_text: str | None = None, source: str = "cli") -> int:
    if once_text is not None:
        answer, _ = _process_turn(container, once_text, source=source)
        print(answer)
        return 0

    print("MMis CLI mode. Commands: /exit, /quit")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            return 0

        answer, _ = _process_turn(container, text, source=source)
        print(f"mmis> {answer}")


def run_voice(container: AppContainer, *, once_text: str | None = None, audio_file: str | None = None) -> int:
    if container.voice is None:
        LOGGER.warning("Voice module is disabled in settings; fallback to CLI mode.")
        return run_cli(container, once_text=once_text, source="voice")

    if audio_file:
        stt = container.voice.transcribe(audio_file)
        text = str(stt.text or "").strip()
        if not text:
            print("voice> STT returned empty text.")
            return 1
        answer, _ = _process_turn(container, text, source="voice")
        print(f"mmis> {answer}")
        container.voice.enqueue_speak(answer)
        return 0

    if once_text is not None:
        answer, _ = _process_turn(container, once_text, source="voice")
        print(answer)
        container.voice.enqueue_speak(answer)
        return 0

    print("MMis voice mode (text input + TTS output). Commands: /exit, /quit")
    while True:
        try:
            text = input("voice> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            return 0
        answer, _ = _process_turn(container, text, source="voice")
        print(f"mmis> {answer}")
        container.voice.enqueue_speak(answer)


def run_ui() -> int:
    from ui.app import main as run_ui_main

    run_ui_main()
    return 0


def run_api(settings: AppSettings) -> int:
    import uvicorn

    uvicorn.run("api.app:app", host=settings.host, port=settings.port, reload=False)
    return 0


def _process_turn(container: AppContainer, text: str, *, source: str) -> tuple[str, Any]:
    user_text = str(text or "").strip()
    if not user_text:
        return "", None

    req_meta = _brain_meta(container=container, source=source)
    result = container.brain.handle_message(user_text, meta=req_meta)
    answer = str(result.text or "").strip()
    if not answer:
        answer = "Пустой ответ от модели."
    return answer, result


def _brain_meta(container: AppContainer, *, source: str) -> dict[str, Any]:
    settings = container.settings
    profile = container.profile
    meta: dict[str, Any] = {
        "source": source,
        "model": settings.model_name,
        "think": bool(settings.thinking_enabled),
        "quality_profile": _state_profile_name(settings.active_profile),
        "temperature": float(profile.generation.temperature),
        "top_p": float(profile.generation.top_p),
        "repeat_penalty": float(profile.generation.repeat_penalty),
    }
    if profile.generation.max_tokens is not None:
        meta["max_tokens"] = int(profile.generation.max_tokens)
    if profile.generation.stop:
        meta["stop"] = list(profile.generation.stop)

    if _resolve_provider_name(settings.llm_default_provider) == "ollama":
        meta.update(
            {
                "num_thread": int(profile.ollama.num_thread),
                "num_ctx": int(profile.ollama.num_ctx),
                "num_gpu": int(profile.ollama.num_gpu),
                "num_batch": int(profile.ollama.num_batch),
                "keep_alive": str(profile.ollama.keep_alive),
            }
        )
    return meta


def _build_automation(settings: AppSettings, event_store: EventStore) -> TaskExecutor:
    allow_actions = settings.safety_mode == "allow_os_actions"
    os_actions = OSActions(
        config=OSActionConfig(
            safe_root=BASE_DIR,
            max_read_kb=1024,
            max_write_kb=1024,
            command_timeout_s=30,
            allow_shell=allow_actions,
            allow_input=allow_actions,
            allow_outside_read=allow_actions,
            allow_outside_write=allow_actions,
        )
    )
    browser = BrowserController(
        os_actions=os_actions,
        config=BrowserConfig(
            read_only=not allow_actions,
            allow_domains=[],
            require_confirmation_for_sensitive=True,
        ),
    )
    return TaskExecutor(browser=browser, os_actions=os_actions, event_store=event_store)


def _state_profile_name(name: str) -> str:
    key = str(name or "BALANCED").strip().upper()
    if key == "ECONOM":
        return "FAST"
    if key in {"FAST", "BALANCED", "QUALITY"}:
        return key
    return "BALANCED"


def _resolve_provider_name(value: str) -> str:
    name = str(value or "ollama").strip().lower()
    if name == "auto":
        return "ollama"
    if name == "openai":
        return "openai"
    return "ollama"


def _safe_call(obj: Any, method_name: str) -> None:
    if obj is None:
        return
    method = getattr(obj, method_name, None)
    if not callable(method):
        return
    try:
        method()
    except Exception:
        LOGGER.debug("shutdown skip: %s.%s", type(obj).__name__, method_name, exc_info=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMis bootstrap launcher")
    parser.add_argument("--mode", choices=["api", "ui", "cli", "voice"], default=None)
    parser.add_argument("--ui", action="store_true")
    parser.add_argument("--cli", action="store_true")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--once", default="", help="Single prompt for CLI/voice mode, then exit")
    parser.add_argument("--audio-file", default="", help="Audio file for voice mode STT (single turn)")
    return parser


def _resolve_mode(args: argparse.Namespace, settings: AppSettings) -> str:
    if bool(args.ui):
        return "ui"
    if bool(args.cli):
        return "cli"
    if bool(args.voice):
        return "voice"
    if bool(args.api):
        return "api"
    if args.mode:
        return str(args.mode).strip().lower()
    start_mode = str(settings.startup_mode or "").strip().lower()
    if start_mode in {"ui", "cli", "voice", "api"}:
        return start_mode
    return "api"


def main() -> None:
    args = _build_arg_parser().parse_args()
    settings = load_config()
    ensure_dirs(memory_dir=settings.memory_dir)
    setup_logging(settings)

    mode = _resolve_mode(args, settings)
    LOGGER.info("starting mode=%s provider=%s profile=%s", mode, settings.llm_default_provider, settings.active_profile)

    if mode == "ui":
        raise SystemExit(run_ui())
    if mode == "api":
        raise SystemExit(run_api(settings))

    container = build_container(settings)
    try:
        if mode == "voice":
            code = run_voice(
                container,
                once_text=(str(args.once).strip() or None),
                audio_file=(str(args.audio_file).strip() or None),
            )
        else:
            code = run_cli(container, once_text=(str(args.once).strip() or None))
    finally:
        container.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
