from memory.short_memory import ShortMemory
from memory.long_memory import LongMemory
from memory.memory_manager import MemoryManager
from memory.user_profile import UserProfile
from brain import Brain
from config import SHORT_MEMORY_LIMIT

short = ShortMemory(limit=SHORT_MEMORY_LIMIT)
longm = LongMemory(path="chroma_db")
profile = UserProfile("user_profile.json")
mm = MemoryManager(short, longm, distance_threshold=0.65)
brain = Brain(mm)

print("Ассистент с долговременной памятью запущен.\n")

while True:
    user_input = input("Ты: ")
    if user_input.lower() == "exit":
        break
    print("AI:", brain.think(user_input))