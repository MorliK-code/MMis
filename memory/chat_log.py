import json
from datetime import datetime

class ChatLog:
    def __init__(self, path: str = "chat_log.jsonl"):
        self.path = path
    
    def append(self, role: str, content: str):
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")