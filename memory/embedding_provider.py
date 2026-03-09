from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    token = []
    for ch in str(text or "").lower():
        if ch.isalnum() or ch == "_":
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def _normalize_vector(values: list[float]) -> list[float]:
    if not values:
        return []
    norm = math.sqrt(sum(float(x) * float(x) for x in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [float(x) / norm for x in values]


class BaseEmbeddingProvider:
    model_name: str = ""
    embedding_version: str = "v2"

    def model_fingerprint(self) -> str:
        raw = f"{self.__class__.__name__}|{self.model_name}|{self.embedding_version}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        batch = self.embed_batch([text])
        return list(batch[0]) if batch else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class HashEmbeddingProvider(BaseEmbeddingProvider):
    dim: int = 384
    model_name: str = "hash_fallback_v2"
    embedding_version: str = "hash_v2"

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        n = max(32, int(self.dim))
        out: list[list[float]] = []
        for text in list(texts or []):
            vec = [0.0] * n
            for token in _tokenize(str(text or "")):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, byteorder="little", signed=False) % n
                vec[idx] += 1.0
            out.append(_normalize_vector(vec))
        return out


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, *, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.model_name = str(model_name or "all-MiniLM-L6-v2").strip()
        self.embedding_version = "sentence_transformers_v1"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("sentence-transformers is not available") from exc
        kwargs: dict[str, Any] = {}
        if str(device or "").strip():
            kwargs["device"] = str(device)
        self._model = SentenceTransformer(self.model_name, **kwargs)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        rows = [str(x or "") for x in list(texts or [])]
        if not rows:
            return []
        # normalize_embeddings gives stable cosine behavior across backends.
        vecs = self._model.encode(rows, normalize_embeddings=True)
        out: list[list[float]] = []
        for row in list(vecs or []):
            if hasattr(row, "tolist"):
                out.append([float(x) for x in row.tolist()])
            else:
                out.append([float(x) for x in list(row or [])])
        return out


class CachedEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, *, base: BaseEmbeddingProvider, cache_path: str | Path):
        self._base = base
        self.model_name = base.model_name
        self.embedding_version = base.embedding_version
        self._path = Path(cache_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                cache_key TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def model_fingerprint(self) -> str:
        return self._base.model_fingerprint()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def _key_for(self, text: str) -> str:
        raw = f"{self.model_name}|{self.embedding_version}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        rows = [str(x or "") for x in list(texts or [])]
        if not rows:
            return []

        keys = [self._key_for(text) for text in rows]
        cached: dict[str, list[float]] = {}
        with self._lock:
            for key in keys:
                row = self._conn.execute(
                    "SELECT embedding_json FROM embeddings_cache WHERE cache_key=?",
                    (key,),
                ).fetchone()
                if not row:
                    continue
                try:
                    cached[key] = [float(x) for x in json.loads(str(row[0] or "[]"))]
                except Exception:
                    continue

        missing_idx = [idx for idx, key in enumerate(keys) if key not in cached]
        if missing_idx:
            missing_texts = [rows[idx] for idx in missing_idx]
            new_vectors = self._base.embed_batch(missing_texts)
            with self._lock:
                for offset, idx in enumerate(missing_idx):
                    key = keys[idx]
                    vector = list(new_vectors[offset]) if offset < len(new_vectors) else []
                    cached[key] = vector
                    self._conn.execute(
                        "INSERT OR REPLACE INTO embeddings_cache(cache_key, embedding_json, created_at) VALUES (?, ?, strftime('%s','now'))",
                        (key, json.dumps(vector, ensure_ascii=False)),
                    )
                self._conn.commit()

        return [list(cached.get(key) or []) for key in keys]


class FallbackEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, *, primary: BaseEmbeddingProvider, fallback: BaseEmbeddingProvider):
        self._primary = primary
        self._fallback = fallback
        self.model_name = primary.model_name
        self.embedding_version = primary.embedding_version

    def model_fingerprint(self) -> str:
        return self._primary.model_fingerprint()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._primary.embed_batch(texts)
        except Exception:
            return self._fallback.embed_batch(texts)


def build_embedding_provider(
    *,
    backend: str,
    model_name: str,
    cache_path: str | Path,
    dim: int = 384,
    device: str | None = None,
) -> BaseEmbeddingProvider:
    backend_name = str(backend or "sentence_transformers").strip().lower()
    fallback = HashEmbeddingProvider(dim=max(32, int(dim)))

    if backend_name in {"sentence_transformers", "sentence-transformers", "st"}:
        try:
            primary: BaseEmbeddingProvider = SentenceTransformerEmbeddingProvider(model_name=model_name, device=device)
        except Exception:
            primary = fallback
    elif backend_name in {"hash", "fallback"}:
        primary = fallback
    else:
        primary = fallback

    provider = FallbackEmbeddingProvider(primary=primary, fallback=fallback)
    return CachedEmbeddingProvider(base=provider, cache_path=cache_path)
