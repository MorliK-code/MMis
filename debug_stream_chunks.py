"""Debug script to check what chunks are received from API streaming."""

import json
import sys
from urllib import request as urllib_request

# Подключаемся к локальному API
API_URL = "http://localhost:8000"

def test_stream_chunks():
    """Проверка, какие чанки реально приходят от API."""
    payload = {
        "text": "Привет, как дела?",
        "store_turn": False,
        "think": True,
    }
    
    req = urllib_request.Request(
        f"{API_URL}/chat/stream",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    
    print("=== Starting stream ===")
    chunk_count = 0
    total_answer_chars = 0
    total_thinking_chars = 0
    
    try:
        with urllib_request.urlopen(req, timeout=120.0) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                
                event = json.loads(line)
                kind = event.get("event", "unknown")
                data = event.get("data", "")
                
                chunk_count += 1
                
                if kind == "chunk":
                    total_answer_chars += len(str(data or ""))
                    print(f"[{chunk_count}] CHUNK ({len(data)} chars): {repr(data[:50])}...")
                elif kind == "thinking":
                    total_thinking_chars += len(str(data or ""))
                    print(f"[{chunk_count}] THINKING ({len(data)} chars): {repr(data[:50])}...")
                elif kind == "final":
                    print(f"[{chunk_count}] FINAL: answer={len(data.get('answer', ''))} thinking={len(data.get('thinking', ''))}")
                    stats = data.get("stats", {})
                    print(f"       Stats: streamed_answer_chars={stats.get('streamed_answer_chars', 0)}, streamed_thinking_chars={stats.get('streamed_thinking_chars', 0)}")
                elif kind == "error":
                    print(f"[{chunk_count}] ERROR: {data}")
                elif kind == "memory_debug":
                    print(f"[{chunk_count}] MEMORY_DEBUG: {type(data)}")
                    
    except Exception as exc:
        print(f"Error: {exc}")
        return
    
    print("\n=== Summary ===")
    print(f"Total chunks: {chunk_count}")
    print(f"Total answer chars streamed: {total_answer_chars}")
    print(f"Total thinking chars streamed: {total_thinking_chars}")


if __name__ == "__main__":
    test_stream_chunks()
