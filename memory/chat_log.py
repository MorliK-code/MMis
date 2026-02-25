import json
from datetime import datetime
from pathlib import Path
from typing import Union


class ChatLog:
    def __init__(self, path: Union[str, Path] = "chat_log.jsonl"):
        self.path = Path(path)

    def append(self, role: str, content: str):
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
