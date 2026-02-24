import ollama
from config import MODEL_NAME

SYSTEM_PROMPT = (
    "Ты — персональный ассистент. Отвечай по-русски, кратко и по делу. "
    "Если пользователь проверяет память — объясняй, что ты запомнил и откуда."
)

class Brain:
    def __init__(self, memory_manager):
        self.mm = memory_manager

    def think(self, user_input: str) -> str:
        recalled = self.mm.recall(user_input, n_results=5)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "system", "content": self.mm.profile.get_summary()})

        if recalled:
            memories_block = "\n\n".join(recalled[:5])
            messages.append({
                "role": "system",
                "content": "Релевантные воспоминания (используй только если реально помогают):\n" + memories_block
            })

        messages.extend(self.mm.short.get())
        
        messages.append({"role": "user", "content": user_input})

        resp = ollama.chat(model=MODEL_NAME, messages=messages)
        reply = resp["message"]["content"]

        self.mm.store_turn(user_input, reply)
        return reply