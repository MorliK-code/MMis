"""
Тесты на ProfileStabilizer — слой медленной эволюции личности.
"""

import unittest
from modules.character.profile_stabilizer import ProfileStabilizer, StabilizerDecision
from modules.character.storage import _default_persona_state, _normalize_persona_state_payload


class ProfileStabilizerTests(unittest.TestCase):
    def test_stabilizer_does_not_promote_transient_irritation_to_baseline(self):
        """
        Тест: transient mood не пишет baseline.
        
        Проверяет, что локальное снижение teasing/sarcasm не уходит в persistent baseline.
        """
        stabilizer = ProfileStabilizer()
        
        current_persona_state = _default_persona_state()
        current_identity_core = {}
        active_profile_snapshot = {
            "interaction_style": {
                "prefers_short_answers": False,  # Временное состояние
            },
        }
        
        decision = stabilizer.decide(
            current_persona_state=current_persona_state,
            current_identity_core=current_identity_core,
            active_profile_snapshot=active_profile_snapshot,
        )
        
        # Недостаточно подтверждений — должно быть rejected или keep_runtime_only
        # (зависит от promotion_class сигнала)
        self.assertEqual(len(decision.promote_to_identity_core), 0)
        self.assertEqual(len(decision.promote_to_persona_baseline), 0)

    def test_stabilizer_promotes_repeated_short_answer_preference_to_identity_core(self):
        """
        Тест: repeated short-answer preference уходит в identity_core.
        
        Проверяет, что после 3+ подтверждений preference продвигается в identity_core.
        """
        stabilizer = ProfileStabilizer()
        
        # Симулируем counters с 3 подтверждениями
        current_persona_state = _default_persona_state()
        current_persona_state["stabilizer"]["counters"] = {
            "interaction_style.prefers_short_answers": {
                "positive": 4,
                "negative": 0,
                "last_seen": "2026-03-24T10:00:00",
            }
        }
        
        current_identity_core = {}
        active_profile_snapshot = {
            "interaction_style": {
                "prefers_short_answers": True,
            },
        }
        
        decision = stabilizer.decide(
            current_persona_state=current_persona_state,
            current_identity_core=current_identity_core,
            active_profile_snapshot=active_profile_snapshot,
        )
        
        # Должно быть promoted в identity_core
        # Проверяем debug информацию
        self.assertGreater(decision.debug.get("promoted_to_identity_core", 0), 0)

    def test_stabilizer_promotes_examples_on_user_code_preference(self):
        """
        Тест: code-example preference закрепляется.
        
        Проверяет, что после 2-3 подтверждений preference продвигается (приоритет высокий).
        """
        stabilizer = ProfileStabilizer()
        
        # Симулируем counters с 2 подтверждениями (высокий приоритет)
        current_persona_state = _default_persona_state()
        current_persona_state["stabilizer"]["counters"] = {
            "interaction_style.prefers_examples_on_user_code": {
                "positive": 2,
                "negative": 0,
                "last_seen": "2026-03-24T10:00:00",
            }
        }
        
        current_identity_core = {}
        active_profile_snapshot = {
            "interaction_style": {
                "prefers_examples_on_user_code": True,
            },
        }
        
        decision = stabilizer.decide(
            current_persona_state=current_persona_state,
            current_identity_core=current_identity_core,
            active_profile_snapshot=active_profile_snapshot,
        )
        
        # Должно быть promoted (высокий приоритет требует меньше подтверждений)
        self.assertGreater(decision.debug.get("promoted_to_identity_core", 0), 0)

    def test_stabilizer_updates_trait_baseline_gradually(self):
        """
        Тест: baseline drift идёт медленно.
        
        Проверяет, что numeric baseline обновляется через rolling window.
        """
        stabilizer = ProfileStabilizer()
        
        # Симулируем counters с 5 наблюдениями
        current_persona_state = _default_persona_state()
        current_persona_state["stabilizer"]["counters"] = {
            "assistant_trait_baseline.directness_baseline": {
                "samples": [0.74, 0.78, 0.76, 0.79, 0.77],
                "last_seen": "2026-03-24T10:00:00",
            }
        }
        
        current_identity_core = {}
        active_profile_snapshot = {
            "assistant_trait_baseline": {
                "directness_baseline": 0.77,
            },
        }
        
        decision = stabilizer.decide(
            current_persona_state=current_persona_state,
            current_identity_core=current_identity_core,
            active_profile_snapshot=active_profile_snapshot,
        )
        
        # Должно быть promoted в persona baseline
        self.assertGreater(decision.debug.get("promoted_to_baseline", 0), 0)
        self.assertIn("directness_baseline", decision.promote_to_persona_baseline)
        
        # Проверяем smoothing (среднее ~0.768)
        avg_value = sum([0.74, 0.78, 0.76, 0.79, 0.77]) / 5
        self.assertAlmostEqual(
            decision.promote_to_persona_baseline["directness_baseline"],
            avg_value,
            places=1
        )

    def test_stabilizer_keeps_existing_identity_when_new_signal_is_weak_or_conflicting(self):
        """
        Тест: conflict не перетирает identity_core сразу.
        
        Проверяет, что слабый сигнал не перетирает существующий identity.
        """
        stabilizer = ProfileStabilizer()
        
        # Симулируем counters с 1 подтверждением (недостаточно)
        current_persona_state = _default_persona_state()
        current_persona_state["stabilizer"]["counters"] = {
            "addressing.canonical_name": {
                "positive": 1,
                "negative": 0,
                "last_seen": "2026-03-24T10:00:00",
            }
        }
        
        current_identity_core = {
            "addressing": {
                "canonical_name": "Паша",  # Существующее имя
            },
        }
        active_profile_snapshot = {
            "addressing": {
                "canonical_name": "Павел",  # Новый сигнал
            },
        }
        
        decision = stabilizer.decide(
            current_persona_state=current_persona_state,
            current_identity_core=current_identity_core,
            active_profile_snapshot=active_profile_snapshot,
        )
        
        # Должно быть rejected (недостаточно подтверждений)
        self.assertEqual(len(decision.promote_to_identity_core), 0)
        self.assertGreater(len(decision.rejected), 0)

    def test_default_persona_state_has_stabilizer(self):
        """
        Тест: default persona state имеет stabilizer.
        """
        state = _default_persona_state()
        
        self.assertIn("stabilizer", state)
        self.assertIn("counters", state["stabilizer"])
        self.assertIn("last_promotion_at", state["stabilizer"])
        self.assertIn("last_decay_at", state["stabilizer"])

    def test_normalize_persona_state_payload_normalizes_stabilizer(self):
        """
        Тест: нормализация persona state сохраняет stabilizer.
        """
        payload = _normalize_persona_state_payload({
            "stabilizer": {
                "counters": {
                    "interaction_style.prefers_short_answers": {
                        "positive": 2,
                        "negative": 0,
                        "last_seen": "2026-03-24T10:00:00",
                    }
                },
                "last_promotion_at": "2026-03-24T09:00:00",
            }
        })
        
        self.assertIn("stabilizer", payload)
        self.assertEqual(payload["stabilizer"]["counters"]["interaction_style.prefers_short_answers"]["positive"], 2)

    def test_stabilizer_decay_counters_removes_old_counters(self):
        """
        Тест: decay counters удаляет старые counters.
        """
        stabilizer = ProfileStabilizer()
        
        counters = {
            "interaction_style.prefers_short_answers": {
                "positive": 2,
                "last_seen": "2026-01-01T10:00:00",  # Старый counter
            },
            "interaction_style.prefers_examples": {
                "positive": 2,
                "last_seen": "2026-03-24T10:00:00",  # Свежий counter
            },
        }
        
        cleaned = stabilizer.decay_counters(counters, max_age_days=30)
        
        # Старый counter должен быть удалён
        self.assertNotIn("interaction_style.prefers_short_answers", cleaned)
        # Свежий counter должен остаться
        self.assertIn("interaction_style.prefers_examples", cleaned)


if __name__ == "__main__":
    unittest.main()
