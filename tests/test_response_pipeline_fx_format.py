from __future__ import annotations

import unittest

from core.response_pipeline import _apply_web_fx_response_guard


class ResponsePipelineFxFormatTests(unittest.TestCase):
    def test_fx_rate_response_is_reformatted_to_compact_web_backed_output(self) -> None:
        fixed, changed, debug = _apply_web_fx_response_guard(
            "I do not have internet access, but maybe the rate is around 41.25.",
            meta={"web_used": True},
            web_evidence_context={
                "query": "usd and eur to uah today",
                "key_facts": [
                    "USD/UAH exchange rate today: 41.25 UAH per USD.",
                    "EUR/UAH exchange rate today: 45.10 UAH per EUR.",
                ],
                "compact_citations": ["minfin.com.ua (2026-03-12)"],
                "selected_rate_type": "cash_rate",
                "final_factual_confidence": 0.82,
                "cautious_synthesis": False,
            },
            web_intent="fx_rate",
        )

        self.assertTrue(changed)
        self.assertEqual(debug["reason"], "fx_compact_format")
        self.assertIn("USD/UAH: 41.25 UAH", fixed)
        self.assertIn("EUR/UAH: 45.10 UAH", fixed)
        self.assertIn("\u0422\u0438\u043f \u043a\u0443\u0440\u0441\u0430: \u043d\u0430\u043b\u0438\u0447\u043d\u044b\u0439", fixed)
        self.assertIn("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: minfin.com.ua (2026-03-12)", fixed)
        self.assertNotIn("internet access", fixed.lower())

    def test_fx_rate_cautious_output_does_not_claim_exact_number(self) -> None:
        fixed, changed, debug = _apply_web_fx_response_guard(
            "The exact USD/UAH rate is 58.56.",
            meta={"web_used": True},
            web_evidence_context={
                "query": "usd to uah today",
                "key_facts": [
                    "USD/UAH exchange rate today: 41.25 UAH per USD.",
                ],
                "compact_citations": ["finance.ua (2026-03-12)"],
                "selected_rate_type": "currency_overview",
                "final_factual_confidence": 0.41,
                "cautious_synthesis": True,
                "conflict_reason": "true_numeric_conflict",
            },
            web_intent="fx_rate",
        )

        self.assertTrue(changed)
        self.assertTrue(debug["cautious"])
        self.assertIn(
            "USD/UAH: \u0442\u043e\u0447\u043d\u043e\u0435 \u0437\u043d\u0430\u0447\u0435\u043d\u0438\u0435 "
            "\u043d\u0430\u0434\u0435\u0436\u043d\u043e \u043d\u0435 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u043e; "
            "\u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0438 \u0440\u0430\u0441\u0445\u043e\u0434\u044f\u0442\u0441\u044f",
            fixed,
        )
        self.assertIn(
            "\u0422\u0438\u043f \u043a\u0443\u0440\u0441\u0430: \u0440\u044b\u043d\u043e\u0447\u043d\u044b\u0439 \u043e\u0431\u0437\u043e\u0440",
            fixed,
        )
        self.assertIn("\u0418\u0441\u0442\u043e\u0447\u043d\u0438\u043a: finance.ua (2026-03-12)", fixed)
        self.assertNotIn("58.56", fixed)


if __name__ == "__main__":
    unittest.main()
