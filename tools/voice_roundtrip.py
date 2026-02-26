from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from brain import Brain
from config import (
    MMIS_VOICE_INPUT_DIR,
    MMIS_VOICE_OUTPUT_DIR,
    MemoryStorageDir,
    SHORT_MEMORY_LIMIT,
)
from memory.assistant_profile import AssistantProfile
from memory.chat_log import ChatLog
from memory.event_store import EventStore
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.short_memory import ShortMemory
from memory.user_profile import UserProfile
from voice import build_voice_module


def build_brain() -> Brain:
    short = ShortMemory(limit=SHORT_MEMORY_LIMIT)
    longm = LongMemory(path=MemoryStorageDir / "chroma_db")
    profile_user = UserProfile(MemoryStorageDir / "user_profile.json")
    profile_assistant = AssistantProfile(MemoryStorageDir / "assistant_profile.json")
    events = EventStore(MemoryStorageDir / "events.json")
    log = ChatLog(MemoryStorageDir / "chat_log.jsonl")
    mm = MemoryManager(short, longm, profile_user, profile_assistant, events, log, distance_threshold=0.65)
    return Brain(mm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice roundtrip: audio -> text -> answer -> audio")
    parser.add_argument("audio_in", help="Input file name in voice/input or full path")
    parser.add_argument(
        "--audio-out",
        default="",
        help="Output file name in voice/output or full path (default: auto-generated in output)",
    )
    parser.add_argument("--no-store", action="store_true", help="Do not save turn to memory")
    return parser.parse_args()


def resolve_audio_in(value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute() or p.parent != Path("."):
        return p
    return MMIS_VOICE_INPUT_DIR / p.name


def resolve_audio_out(value: str) -> Path:
    p = Path(value).expanduser() if value else Path()
    if not value:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return MMIS_VOICE_OUTPUT_DIR / f"reply_{ts}.mp3"
    if p.is_absolute() or p.parent != Path("."):
        return p
    return MMIS_VOICE_OUTPUT_DIR / p.name


def main() -> int:
    args = parse_args()
    audio_in = resolve_audio_in(args.audio_in)
    audio_out = resolve_audio_out(args.audio_out)

    voice = build_voice_module()
    brain = build_brain()

    user_text = voice.speech_to_text(audio_in)
    if not user_text:
        print("STT result is empty. Check microphone recording or STT backend settings.")
        return 2

    print(f"USER (from voice): {user_text}")
    reply = brain.think(user_text, store_turn=not args.no_store)
    print(f"AI: {reply}")

    result_path = voice.text_to_speech(reply, audio_out)
    print(f"Saved TTS audio to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
