from __future__ import annotations

from _output_utils import enable_unittest_json_output

enable_unittest_json_output()

import unittest

from core.spec_registry import load_spec, validate_no_txt_paths


class SpecRegistryTests(unittest.TestCase):
    def test_load_core_specs(self) -> None:
        taxonomy = load_spec("taxonomy", required=True)
        output = load_spec("output", required=True)
        self.assertIn("langs", taxonomy)
        self.assertIn("order", output)

    def test_validate_rejects_txt_prompt_path(self) -> None:
        with self.assertRaises(ValueError):
            validate_no_txt_paths({"prompt_template_path": "prompts/system/base_system.txt"})


if __name__ == "__main__":
    unittest.main()

