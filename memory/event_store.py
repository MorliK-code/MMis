import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class EventStore:
    def __init__(self, path: str = "events.json"):
        self.path = path
        self.events: List[Dict[str, Any]] = []
        self.load()

    def load(self):
        if os.path.exists(self.path):
           with open(self.path, "r", encoding="utf-8") as f:
                self.events = json.load(f)
        else:
            self.events = []

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=False, indent=2)
    
    def add(self, event: Dict[str, Any]):
        event = dict(event)
        event["ts"] = event.get("ts") or datetime.now().isoformat(timespec="seconds")
        self.events.append(event)
        self.save()

    def last(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.events[-n:]
    
    def last_of_type(self, event_type: str) -> Optional[Dict[str, Any]]:
        for e in reversed(self.events):
            if e.get("type") == event_type:
                return e
        return None