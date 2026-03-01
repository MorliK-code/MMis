from __future__ import annotations

from _output_utils import enable_unittest_json_output
enable_unittest_json_output()

import unittest

from prompt_engine.prompt_registry import PromptRegistry


class PromptRegistryMetadataTests(unittest.TestCase):
    def test_frontmatter_is_parsed(self) -> None:
        reg = PromptRegistry()
        doc = reg.get_prompt("system.base", use_cache=False)

        self.assertEqual(doc.id, "base_system")
        self.assertRegex(str(doc.version), r"^\d+\.\d+\.\d+$")
        self.assertIn("core", doc.tags)
        self.assertGreaterEqual(int(doc.min_ctx), 1)
        self.assertNotIn("---", doc.text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()

