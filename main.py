from memory.short_memory import ShortMemory
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.user_profile import UserProfile
from memory.event_store import EventStore
from memory.chat_log import ChatLog
from brain import Brain
from config import SHORT_MEMORY_LIMIT
from config import MemoryStorageDir
from memory.assistant_profile import AssistantProfile

short = ShortMemory(limit=SHORT_MEMORY_LIMIT)
longm = LongMemory(path=MemoryStorageDir / "chroma_db")
profile_user = UserProfile(MemoryStorageDir / "user_profile.json")
profile_assistant = AssistantProfile(MemoryStorageDir / "assistant_profile.json")
events = EventStore(MemoryStorageDir / "events.json")
log = ChatLog(MemoryStorageDir / "chat_log.jsonl")

mm = MemoryManager(short, longm, profile_user, profile_assistant, events, log, distance_threshold=0.65)
brain = Brain(mm)

print("MMis запущен.\n")

while True:
    user_input = input()
    if user_input.lower() == "exit":
        break
    print("AI:", brain.think(user_input))