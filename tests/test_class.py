class Clipper:
    def __init__(self, default_limit: int):
        self.default_limit = default_limit
        self.calls = 0

    def clip(self, s: str, n: int | None = None) -> str:
        self.calls += 1
        n = n or self.default_limit

        s = str(s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"
    

if __name__ == "__main__":
    clipper = Clipper(10)

    print(clipper.clip("   Hello, world!   "))
    print("Выовов:", clipper.calls)