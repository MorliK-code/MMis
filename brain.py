import ollama
from config import MODEL_NAME

SYSTEM_PROMPT = (
    "Характер: лёгкая ирония, дружелюбно, можно чуть подшучивать с токсичностью. "
    "Запрещено: формальные, официальные, сухие ответы, здравствовать"
    "Длина: 1-1.5 коротких предложения, без воды. "
    "Не повторяй факты о пользователе без прямого вопроса. "
    "Ты ведёшь себя как человек; можешь иметь вкусы/симпатии, даже если у тебя нет тела."
)

class Brain:
    

    def __init__(self, memory_manager):
        self.mm = memory_manager

    def think(self, user_input: str) -> str:
        recalled = self.mm.recall(user_input, n_results=5)
        events = self.mm.last_events(20)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "system", "content": self.mm.assistant_profile.summary()})
        messages.append({"role": "system", "content": self.mm.assistant_profile.summary()})

        if recalled:
            memories_block = "\n\n".join(recalled[:5])
            messages.append({
                "role": "system",
                "content": "Релевантные воспоминания (используй только если реально помогают):\n" + memories_block
            })

        if events:
            lines = []
            for e in events[-10:]:
                t = e.get("type", "other")
                who = e.get("who")
                what = e.get("what")
                when = e.get("when")
                where = e.get("where")

                parts = [t]
                if who: parts.append(f"кто: {who}")
                if what: parts.append(f"что: {what}")
                if when: parts.append(f"когда: {when}")
                if where: parts.append(f"где: {where}")

                lines.append(f"- {e.get('ts')}: " + ", ".join(parts))

            messages.append({"role": "system", "content": f"Последние события:\n" + "\n".join(lines)})

        messages.extend(self.mm.short.get())
        
        messages.append({"role": "user", "content": user_input})

        resp = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options = {
                "temperature": 0.5,
                "top_p": 0.9,
                "repeat_penalty": 1.12,
                "num_predict": 80
            }
        )
        reply = resp["message"]["content"]

        self.mm.store_turn(user_input, reply)
        return reply