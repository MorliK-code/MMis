from memory.fact_extractor import extract_facts

class MemoryManager:
    def __init__(self, short_memory, long_memory, distance_threshold=0.65):
        self.short = short_memory
        self.long = long_memory
        self.distance_threshold = distance_threshold

    def store_turn(self, user_text: str, assistant_text: str):
        self.short.add("user", user_text)
        self.short.add("assistant", assistant_text)

        combined = f"User: {user_text}\nAssistant: {assistant_text}"
        self.long.add(combined, meta={"type": "dialog_turn"})

        facts = extract_facts(user_text, assistant_text)
        for k, v in facts.items():
            self.profile.upsert(k, v)

    def recall(self, user_text: str, n_results: int = 5):
        found = self.long.search(user_text, n_results=n_results)
        filtered = [doc for (doc, dist, _meta) in found if dist is not None and dist <= self.distance_threshold]
        return filtered