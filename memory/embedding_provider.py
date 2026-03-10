"""Embedding provider adapters for Memory V2 foundation."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory.memory_models import EmbeddingProvider

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


class BaseEmbeddingProvider(EmbeddingProvider):
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
        if "jina" in self.model_name.lower():
            kwargs["trust_remote_code"] = True
        try:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except TypeError:
            # Older sentence-transformers builds may not support trust_remote_code.
            kwargs.pop("trust_remote_code", None)
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


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str = "hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M",
        host: str | None = None,
        timeout_sec: float | None = None,
    ):
        self.model_name = str(model_name or "hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M").strip()
        self.embedding_version = "ollama_embedding_v1"
        self._host = str(host or "").strip()
        try:
            import ollama  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ollama package is not available") from exc

        kwargs: dict[str, Any] = {}
        if self._host:
            kwargs["host"] = self._host
        if timeout_sec is not None:
            kwargs["timeout"] = float(timeout_sec)
        self._client = ollama.Client(**kwargs)

    @staticmethod
    def _coerce_vectors(payload: Any) -> list[list[float]]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get("embeddings")
        if isinstance(rows, list):
            out: list[list[float]] = []
            for row in rows:
                if isinstance(row, list):
                    out.append([float(x) for x in row])
            if out:
                return out
        one = payload.get("embedding")
        if isinstance(one, list):
            return [[float(x) for x in one]]
        return []

    def _model_candidates(self) -> list[str]:
        primary = str(self.model_name or "").strip()
        if not primary:
            return []
        out = [primary]
        low = primary.lower()
        # Convenience alias: allow local short model names for HF Qwen GGUF pulls.
        if "/" not in primary and low.startswith("qwen") and "gguf" in low:
            out.append(f"hf.co/Qwen/{primary}")
        return out

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        rows = [str(x or "") for x in list(texts or [])]
        if not rows:
            return []

        last_error: Exception | None = None
        for model_name in self._model_candidates():
            # Newer Ollama Python clients: client.embed(model=..., input=[...]).
            embed = getattr(self._client, "embed", None)
            if callable(embed):
                try:
                    payload = embed(model=model_name, input=rows)
                    vecs = self._coerce_vectors(payload)
                    if vecs:
                        return vecs
                except Exception as exc:
                    last_error = exc

            # Older clients: client.embeddings(model=..., prompt="...") one-by-one.
            embeddings = getattr(self._client, "embeddings", None)
            if callable(embeddings):
                try:
                    out: list[list[float]] = []
                    for text in rows:
                        payload = embeddings(model=model_name, prompt=text)
                        vecs = self._coerce_vectors(payload)
                        out.append(list(vecs[0]) if vecs else [])
                    if out:
                        return out
                except Exception as exc:
                    last_error = exc

        if last_error is not None:
            raise RuntimeError(str(last_error))
        raise RuntimeError("Ollama client does not expose embed/embeddings methods.")


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


class FixedDimEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, *, base: BaseEmbeddingProvider, dim: int):
        self._base = base
        self._dim = max(32, int(dim))
        self.model_name = base.model_name
        self.embedding_version = f"{base.embedding_version}|fixed_dim_{self._dim}"

    def model_fingerprint(self) -> str:
        raw = f"{self._base.model_fingerprint()}|fixed_dim:{self._dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _resize(self, vector: list[float]) -> list[float]:
        src = [float(x) for x in list(vector or [])]
        if not src:
            return [0.0 for _ in range(self._dim)]
        if len(src) == self._dim:
            return _normalize_vector(src)
        if len(src) < self._dim:
            return _normalize_vector(src + ([0.0] * (self._dim - len(src))))
        folded = [0.0] * self._dim
        for idx, value in enumerate(src):
            folded[idx % self._dim] += float(value)
        return _normalize_vector(folded)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        rows = self._base.embed_batch(texts)
        return [self._resize(row) for row in list(rows or [])]


def build_embedding_provider(
    *,
    backend: str,
    model_name: str,
    cache_path: str | Path,
    dim: int = 384,
    device: str | None = None,
    ollama_host: str | None = None,
    ollama_timeout_sec: float | None = None,
) -> BaseEmbeddingProvider:
    target_dim = max(32, int(dim))
    backend_name = str(backend or "sentence_transformers").strip().lower()
    fallback = HashEmbeddingProvider(dim=target_dim)

    if backend_name in {"sentence_transformers", "sentence-transformers", "st"}:
        try:
            primary: BaseEmbeddingProvider = SentenceTransformerEmbeddingProvider(model_name=model_name, device=device)
        except Exception:
            primary = fallback
    elif backend_name in {"hash", "fallback"}:
        primary = fallback
    elif backend_name in {"ollama", "ollama_embeddings", "ollama-embeddings"}:
        try:
            primary = OllamaEmbeddingProvider(
                model_name=model_name,
                host=ollama_host,
                timeout_sec=ollama_timeout_sec,
            )
        except Exception:
            primary = fallback
    else:
        primary = fallback

    provider = FallbackEmbeddingProvider(primary=primary, fallback=fallback)
    provider = FixedDimEmbeddingProvider(base=provider, dim=target_dim)
    return CachedEmbeddingProvider(base=provider, cache_path=cache_path)
