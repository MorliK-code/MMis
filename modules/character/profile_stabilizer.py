"""
Profile Stabilizer — слой медленной эволюции личности.

Отделяет:
- временное настроение (mood)
- состояние сессии (session state)
- устойчивые предпочтения (persistent preferences)
- identity-level параметры (identity core)

Не меняет личность на каждый turn.
Требует нескольких подтверждений перед записью в baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class StabilizerDecision:
    """
    Решение стабилизатора о том, что делать с сигналом.
    """
    # Продвинуть в identity_core
    promote_to_identity_core: dict[str, Any] = field(default_factory=dict)
    
    # Продвинуть в persona baseline
    promote_to_persona_baseline: dict[str, float] = field(default_factory=dict)
    
    # Оставить только в runtime (не сохранять persistent)
    keep_runtime_only: dict[str, Any] = field(default_factory=dict)
    
    # Отклонить (недостаточно подтверждений / конфликт)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    
    # Debug информация
    debug: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Преобразует в словарь."""
        return {
            "promote_to_identity_core": self.promote_to_identity_core,
            "promote_to_persona_baseline": self.promote_to_persona_baseline,
            "keep_runtime_only": self.keep_runtime_only,
            "rejected": self.rejected,
            "debug": self.debug,
        }


class ProfileStabilizer:
    """
    Стабилизатор профиля личности.

    Решает, какие сигналы можно записать в persistent state,
    а какие должны остаться временными.
    """

    # Пороги для promotion
    IDENTITY_CORE_MIN_CONFIRMATIONS = 3
    BASELINE_TRAIT_MIN_CONFIRMATIONS = 3
    BASELINE_TRAIT_WINDOW_SIZE = 5
    
    # Коэффициенты для baseline drift
    BASELINE_DRIFT_SMOOTHING = 0.85  # old_baseline * 0.85 + observed * 0.15
    
    # Время жизни counters (дни)
    COUNTER_TTL_DAYS = 30

    def decide(
        self,
        *,
        current_persona_state: dict[str, Any],
        current_identity_core: dict[str, Any],
        active_profile_snapshot: dict[str, Any],
        memory_reasoning_snapshot: dict[str, Any] | None = None,
        session_stats: dict[str, Any] | None = None,
    ) -> StabilizerDecision:
        """
        Принимает решение о promotion сигналов.

        Args:
            current_persona_state: Текущее состояние persona (из storage).
            current_identity_core: Текущий identity core.
            active_profile_snapshot: Активный профиль из retrieval.
            memory_reasoning_snapshot: Snapshot из memory reasoning (опционально).
            session_stats: Статистика сессии (опционально).

        Returns:
            StabilizerDecision с решениями о promotion.
        """
        decision = StabilizerDecision(
            debug={
                "identity_core_keys": list(current_identity_core.keys()),
                "persona_state_keys": list(current_persona_state.keys()),
                "profile_snapshot_keys": list(active_profile_snapshot.keys()) if active_profile_snapshot else [],
            }
        )

        # Извлекаем stabilizer counters из persona_state
        stabilizer_state = current_persona_state.get("stabilizer", {})
        counters = dict(stabilizer_state.get("counters", {}))

        # Собираем сигналы из различных источников
        signals = self._collect_signals(
            active_profile_snapshot=active_profile_snapshot,
            memory_reasoning_snapshot=memory_reasoning_snapshot,
            session_stats=session_stats,
        )

        # Обрабатываем каждый сигнал
        for signal in signals:
            signal_result = self._process_signal(
                signal=signal,
                counters=counters,
                current_identity_core=current_identity_core,
            )
            
            if signal_result["action"] == "promote_to_identity_core":
                decision.promote_to_identity_core.update(signal_result.get("patch", {}))
            elif signal_result["action"] == "promote_to_persona_baseline":
                decision.promote_to_persona_baseline.update(signal_result.get("patch", {}))
            elif signal_result["action"] == "keep_runtime_only":
                decision.keep_runtime_only.update(signal_result.get("patch", {}))
            elif signal_result["action"] == "rejected":
                decision.rejected.append(signal_result)

        # Обновляем debug
        decision.debug["signals_processed"] = len(signals)
        decision.debug["promoted_to_identity_core"] = len(decision.promote_to_identity_core)
        decision.debug["promoted_to_baseline"] = len(decision.promote_to_persona_baseline)
        decision.debug["rejected_count"] = len(decision.rejected)

        return decision

    def _collect_signals(
        self,
        active_profile_snapshot: dict[str, Any],
        memory_reasoning_snapshot: dict[str, Any] | None,
        session_stats: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """
        Собирает сигналы из различных источников.

        Returns:
            Список сигналов для обработки.
        """
        signals = []

        # Сигналы из profile snapshot
        if active_profile_snapshot:
            signals.extend(self._extract_signals_from_profile(active_profile_snapshot))

        # Сигналы из memory reasoning
        if memory_reasoning_snapshot:
            signals.extend(self._extract_signals_from_memory(memory_reasoning_snapshot))

        # Сигналы из session stats
        if session_stats:
            signals.extend(self._extract_signals_from_session(session_stats))

        return signals

    def _extract_signals_from_profile(
        self,
        profile_snapshot: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Извлекает сигналы из profile snapshot."""
        signals = []

        # Interaction style signals
        interaction_style = profile_snapshot.get("interaction_style", {})
        
        if "prefers_short_answers" in interaction_style:
            signals.append({
                "type": "interaction_style",
                "key": "prefers_short_answers",
                "value": interaction_style["prefers_short_answers"],
                "source": "profile_snapshot",
                "promotion_class": "identity_core",
            })

        if "prefers_examples_on_user_code" in interaction_style:
            signals.append({
                "type": "interaction_style",
                "key": "prefers_examples_on_user_code",
                "value": interaction_style["prefers_examples_on_user_code"],
                "source": "profile_snapshot",
                "promotion_class": "identity_core",
                "priority": "high",  # Важно для проектного контекста
            })

        if "technical_collaboration_style" in interaction_style:
            signals.append({
                "type": "interaction_style",
                "key": "technical_collaboration_style",
                "value": interaction_style["technical_collaboration_style"],
                "source": "profile_snapshot",
                "promotion_class": "identity_core",
            })

        # Assistant trait baseline signals
        trait_baseline = profile_snapshot.get("assistant_trait_baseline", {})
        
        for trait_key in ["warmth_baseline", "directness_baseline", "empathy_floor", 
                          "professionalism_floor", "sarcasm_ceiling"]:
            if trait_key in trait_baseline:
                signals.append({
                    "type": "assistant_trait_baseline",
                    "key": trait_key,
                    "value": float(trait_baseline[trait_key]),
                    "source": "profile_snapshot",
                    "promotion_class": "persona_baseline",
                })

        # Addressing signals
        addressing = profile_snapshot.get("addressing", {})
        
        if "canonical_name" in addressing and addressing["canonical_name"]:
            signals.append({
                "type": "addressing",
                "key": "canonical_name",
                "value": addressing["canonical_name"],
                "source": "profile_snapshot",
                "promotion_class": "identity_core",
                "requires_strong_confidence": True,
            })

        return signals

    def _extract_signals_from_memory(
        self,
        memory_reasoning_snapshot: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Извлекает сигналы из memory reasoning snapshot."""
        signals = []

        # Пример: preference signals из memory
        preferences = memory_reasoning_snapshot.get("preferences", [])
        for pref in preferences:
            if isinstance(pref, dict):
                signals.append({
                    "type": "preference",
                    "key": pref.get("key", "unknown"),
                    "value": pref.get("value"),
                    "source": "memory_reasoning",
                    "promotion_class": pref.get("promotion_class", "runtime_only"),
                    "confidence": float(pref.get("confidence", 0.5)),
                })

        return signals

    def _extract_signals_from_session(
        self,
        session_stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Извлекает сигналы из session statistics."""
        signals = []

        # Пример: response pattern signals
        avg_response_length = session_stats.get("avg_response_length")
        if avg_response_length is not None:
            # Короткие ответы могут указывать на предпочтение краткости
            if avg_response_length < 100:  # условный порог
                signals.append({
                    "type": "response_pattern",
                    "key": "prefers_short_answers",
                    "value": True,
                    "source": "session_stats",
                    "promotion_class": "identity_core",
                })

        return signals

    def _process_signal(
        self,
        signal: dict[str, Any],
        counters: dict[str, Any],
        current_identity_core: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Обрабатывает отдельный сигнал.

        Returns:
            Результат обработки с action и patch.
        """
        signal_key = f"{signal['type']}.{signal['key']}"
        signal_counter = counters.get(signal_key, {})

        # Проверяем, есть ли уже достаточно подтверждений в counters
        if signal_counter:
            # Проверяем counter для promotion
            if signal.get("promotion_class") == "identity_core":
                result = self._evaluate_identity_core_promotion(
                    signal=signal,
                    counter=signal_counter,
                    current_identity_core=current_identity_core,
                )
                if result["action"] == "promote_to_identity_core":
                    return result
            
            elif signal.get("promotion_class") == "persona_baseline":
                result = self._evaluate_baseline_promotion(
                    signal=signal,
                    counter=signal_counter,
                )
                if result["action"] == "promote_to_persona_baseline":
                    return result

        # Обновляем counter для нового сигнала
        updated_counter = self._update_counter(
            counter=signal_counter,
            signal=signal,
        )

        # Проверяем, достигнут ли порог для promotion
        if signal.get("promotion_class") == "identity_core":
            return self._evaluate_identity_core_promotion(
                signal=signal,
                counter=updated_counter,
                current_identity_core=current_identity_core,
            )
        elif signal.get("promotion_class") == "persona_baseline":
            return self._evaluate_baseline_promotion(
                signal=signal,
                counter=updated_counter,
            )
        else:
            # runtime_only
            return {
                "action": "keep_runtime_only",
                "patch": {signal_key: signal["value"]},
                "reason": "runtime_only_signal",
            }

    def _update_counter(
        self,
        counter: dict[str, Any],
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        """Обновляет counter для сигнала."""
        now = datetime.now().isoformat()
        
        if signal.get("promotion_class") in ("identity_core", "persona_baseline"):
            # Бинарные или categorical сигналы
            if not isinstance(signal.get("value"), (int, float)):
                positive = counter.get("positive", 0)
                negative = counter.get("negative", 0)
                
                # Простая логика: если значение truthy — положительный сигнал
                if signal.get("value"):
                    positive += 1
                else:
                    negative += 1
                
                return {
                    "positive": positive,
                    "negative": negative,
                    "last_seen": now,
                }
            else:
                # Numeric signals — rolling window
                samples = list(counter.get("samples", []))
                samples.append(float(signal["value"]))
                
                # Ограничиваем размер окна
                max_samples = self.BASELINE_TRAIT_WINDOW_SIZE
                if len(samples) > max_samples:
                    samples = samples[-max_samples:]
                
                return {
                    "samples": samples,
                    "last_seen": now,
                }

        return counter

    def _evaluate_identity_core_promotion(
        self,
        signal: dict[str, Any],
        counter: dict[str, Any],
        current_identity_core: dict[str, Any],
    ) -> dict[str, Any]:
        """Оценивает, можно ли продвинуть сигнал в identity_core."""
        # Проверяем порог подтверждений
        min_confirmations = self.IDENTITY_CORE_MIN_CONFIRMATIONS
        
        # Приоритетные сигналы требуют меньше подтверждений
        if signal.get("priority") == "high":
            min_confirmations = 2
        
        # Сигналы с высоким требованием confidence требуют больше подтверждений
        if signal.get("requires_strong_confidence"):
            min_confirmations = 5

        positive = counter.get("positive", 0)
        
        if positive >= min_confirmations:
            # Достигнут порог — продвигаем в identity_core
            patch = {}
            
            if signal["type"] == "interaction_style":
                patch["interaction_style"] = {
                    signal["key"]: signal["value"],
                }
            elif signal["type"] == "addressing":
                patch["addressing"] = {
                    signal["key"]: signal["value"],
                }
            else:
                patch[signal["type"]] = {
                    signal["key"]: signal["value"],
                }

            return {
                "action": "promote_to_identity_core",
                "patch": patch,
                "reason": f"reached_confirmation_threshold ({positive}>={min_confirmations})",
            }
        else:
            # Недостаточно подтверждений
            return {
                "action": "rejected",
                "reason": f"insufficient_confirmations ({positive}<{min_confirmations})",
                "signal": signal,
                "counter": counter,
            }

    def _evaluate_baseline_promotion(
        self,
        signal: dict[str, Any],
        counter: dict[str, Any],
    ) -> dict[str, Any]:
        """Оценивает, можно ли продвинуть сигнал в persona baseline."""
        samples = counter.get("samples", [])
        min_samples = self.BASELINE_TRAIT_MIN_CONFIRMATIONS

        if len(samples) >= min_samples:
            # Вычисляем среднее значение
            avg_value = sum(samples) / len(samples)
            
            # Применяем smoothing
            smoothed_value = avg_value  # В простой версии просто среднее
            
            return {
                "action": "promote_to_persona_baseline",
                "patch": {signal["key"]: smoothed_value},
                "reason": f"reached_sample_threshold ({len(samples)}>={min_samples})",
            }
        else:
            return {
                "action": "rejected",
                "reason": f"insufficient_samples ({len(samples)}<{min_samples})",
                "signal": signal,
                "counter": counter,
            }

    def decay_counters(
        self,
        counters: dict[str, Any],
        *,
        max_age_days: float | None = None,
    ) -> dict[str, Any]:
        """
        Ослабляет старые counters.

        Args:
            counters: Текущие counters.
            max_age_days: Максимальный возраст counter (дни).

        Returns:
            Обновлённые counters.
        """
        if max_age_days is None:
            max_age_days = self.COUNTER_TTL_DAYS

        now = datetime.now()
        cleaned_counters = {}

        for key, counter in counters.items():
            last_seen_str = counter.get("last_seen")
            if not last_seen_str:
                # Сохраняем counters без last_seen
                cleaned_counters[key] = counter
                continue

            try:
                last_seen = datetime.fromisoformat(last_seen_str)
                age_days = (now - last_seen).days

                if age_days <= max_age_days:
                    cleaned_counters[key] = counter
                # else: удаляем старый counter
            except (ValueError, TypeError):
                # Неверный формат даты — сохраняем counter
                cleaned_counters[key] = counter

        return cleaned_counters
