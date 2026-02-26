import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Union


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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", (text or "").lower())]

    def _tail_records(self, max_lines: int = 400) -> List[Dict]:
        if not self.path.exists():
            return []

        buf: deque[str] = deque(maxlen=max(10, int(max_lines)))
        with self.path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    buf.append(line)

        out: List[Dict] = []
        for line in buf:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def last_messages(self, n: int = 30, roles: Iterable[str] = ("user", "assistant")) -> List[Dict]:
        allow = set(roles or ())
        rows = self._tail_records(max_lines=max(120, n * 4))
        if allow:
            rows = [r for r in rows if str(r.get("role", "")).lower() in allow]
        return rows[-n:]

    def relevant_messages(
        self,
        query: str,
        limit: int = 6,
        scan_last: int = 400,
        roles: Iterable[str] = ("user", "assistant"),
    ) -> List[Dict]:
        allow = set(roles or ())
        q_tokens = set(self._tokenize(query))
        if not q_tokens:
            return self.last_messages(n=min(6, limit), roles=roles)

        candidates = self._tail_records(max_lines=scan_last)
        scored = []
        for i, rec in enumerate(candidates):
            role = str(rec.get("role", "")).lower()
            if allow and role not in allow:
                continue
            content = str(rec.get("content", "") or "")
            if not content:
                continue
            toks = set(self._tokenize(content))
            if not toks:
                continue
            overlap = len(q_tokens & toks)
            if overlap <= 0:
                continue
            # Slight recency bias: newer lines win when overlap is equal.
            recency_bonus = i / max(1, len(candidates))
            score = overlap + recency_bonus
            scored.append((score, i, rec))

        if not scored:
            return self.last_messages(n=min(6, limit), roles=roles)

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = [x[2] for x in scored[: max(1, limit)]]
        top.sort(key=lambda r: str(r.get("ts", "")))
        return top
