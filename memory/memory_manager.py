from memory.fact_extractor import extract_facts
from memory.event_exctractor import extract_events_llm

class MemoryManager:
    def __init__(self, short_memory, long_memory, profile, event_store, chat_log, distance_threshold=0.65):
        self.short = short_memory
        self.long = long_memory
        self.profile = profile
        self.events = event_store
        self.log = chat_log
        self.distance_threshold = distance_threshold

    def store_turn(self, user_text: str, assistant_text: str):
        # log all   
        self.log.append("user", user_text)
        self.log.append("assistant", assistant_text)

        # events
        try:
            evs = extract_events_llm(user_text)
        except Exception as e:
            evs = []

        for ev in evs:
            self.events.add(ev)

        # short memory
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        # long memory
        combined = f"User: {user_text}\nAssistant: {assistant_text}"
        self.long.add(combined, meta={"type": "dialog_turn"})

        # profile memory
        facts = extract_facts(user_text, assistant_text)
        for k, v in facts.items():
            self.profile.upsert(k, v)

    def recall(self, user_text: str, n_results: int = 5):
        found = self.long.search(user_text, n_results=n_results)
        return [
            doc for (doc, dist, _meta) in found
            if dist is not None and dist <= self.distance_threshold
        ]
    
    def last_events(self, n: int = 20):
        return self.events.last(n)
    
    def last_events_of_type(self, event_type: str):
        return self.events.last_of_type(event_type)