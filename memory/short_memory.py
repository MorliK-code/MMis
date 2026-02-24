class ShortMemory:
    def __init__(self, limit=20):
        self.limit = limit
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.limit:
            self.messages = self.messages[-self.limit:]

    def get(self):
        return list(self.messages)