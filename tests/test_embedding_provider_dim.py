from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import math
import tempfile
import unittest
from pathlib import Path

from memory.embedding_provider import build_embedding_provider


class EmbeddingProviderDimTests(unittest.TestCase):
    def test_hash_backend_respects_requested_dim(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_embed_dim_") as tmpdir:
            provider = build_embedding_provider(
                backend="hash",
                model_name="hash-only",
                cache_path=Path(tmpdir) / "cache.sqlite3",
                dim=512,
            )
            try:
                vector = provider.embed("dimension check sentence")
                self.assertEqual(len(vector), 512)
                norm = math.sqrt(sum(float(x) * float(x) for x in vector))
                self.assertAlmostEqual(norm, 1.0, places=5)
            finally:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()

    def test_dim_change_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_embed_fp_") as tmpdir:
            base = Path(tmpdir)
            p512 = build_embedding_provider(
                backend="hash",
                model_name="hash-only",
                cache_path=base / "cache_512.sqlite3",
                dim=512,
            )
            p768 = build_embedding_provider(
                backend="hash",
                model_name="hash-only",
                cache_path=base / "cache_768.sqlite3",
                dim=768,
            )
            try:
                self.assertNotEqual(p512.model_fingerprint(), p768.model_fingerprint())
            finally:
                for provider in (p512, p768):
                    close = getattr(provider, "close", None)
                    if callable(close):
                        close()

    def test_ollama_backend_uses_fixed_dim_or_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mmis_embed_ollama_") as tmpdir:
            provider = build_embedding_provider(
                backend="ollama",
                model_name="hf.co/Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M",
                cache_path=Path(tmpdir) / "cache.sqlite3",
                dim=768,
                ollama_host="http://127.0.0.1:11434",
                ollama_timeout_sec=1.0,
            )
            try:
                vector = provider.embed("hello embedding")
                self.assertEqual(len(vector), 768)
                norm = math.sqrt(sum(float(x) * float(x) for x in vector))
                # Non-empty vectors are normalized by provider pipeline.
                if norm > 1e-12:
                    self.assertAlmostEqual(norm, 1.0, places=4)
            finally:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()


if __name__ == "__main__":
    unittest.main()
