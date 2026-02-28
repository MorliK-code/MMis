from __future__ import annotations

import unittest

from prompt_engine.prompt_registry import PromptRegistry


class PromptRegistryMetadataTests(unittest.TestCase):
    def test_frontmatter_is_parsed(self) -> None:
        reg = PromptRegistry()
        doc = reg.get_prompt("system.base", use_cache=False)

        self.assertEqual(doc.id, "base_system")
        self.assertEqual(doc.version, "1.4.2")
        self.assertIn("core", doc.tags)
        self.assertGreaterEqual(int(doc.min_ctx), 1)
        self.assertNotIn("---", doc.text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
