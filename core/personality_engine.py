from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.paths import BASE_DIR, DATA_DIR


DEFAULT_COOLDOWN_SEC = 90.0
DEFAULT_MIN_CONFIDENCE = 0.65
DEFAULT_BLEND_STEPS = 4


@dataclass(frozen=True)
class PersonalityProfile:
    id: str
    name: str
    version: str
    system_prompt: str
    style_prompt: str
    rules_prompt: str
    voice_style: str
    llm_profile: str
    traits: dict[str, float] = field(default_factory=dict)
    triggers: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "style_prompt": self.style_prompt,
            "rules_prompt": self.rules_prompt,
            "voice_style": self.voice_style,
            "llm_profile": self.llm_profile,
            "traits": dict(self.traits or {}),
            "triggers": {
                "auto_enable_intents": list((self.triggers or {}).get("auto_enable_intents") or []),
                "auto_disable_intents": list((self.triggers or {}).get("auto_disable_intents") or []),
            },
        }


@dataclass(frozen=True)
class PersonalityDecision:
    target_personality_id: str
    confidence: float
    reason: str
    switched: bool
    lock_after_switch: bool = False
    blend: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_personality_id": self.target_personality_id,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "switched": bool(self.switched),
            "lock_after_switch": bool(self.lock_after_switch),
            "blend": dict(self.blend or {}),
            "ts": float(self.ts or 0.0),
        }


class PersonalityEngine:
    def __init__(
        self,
        *,
        profiles_dir: str | Path | None = None,
        default_personality_id: str = "default",
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        blend_steps: int = DEFAULT_BLEND_STEPS,
    ):
        self.profiles_dir = Path(profiles_dir).expanduser() if profiles_dir is not None else (DATA_DIR / "personalities")
        self.default_personality_id = str(default_personality_id or "default").strip().lower() or "default"
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.blend_steps = max(1, int(blend_steps))
        self._profiles = self._load_profiles()

    def reload(self) -> None:
        self._profiles = self._load_profiles()

    def list_ids(self) -> list[str]:
        return sorted(self._profiles.keys())

    def get_profile(self, profile_id: str) -> PersonalityProfile:
        key = str(profile_id or "").strip().lower()
        if key in self._profiles:
            return self._profiles[key]
        return self._profiles[self.default_personality_id]

    def decide(
        self,
        *,
        intent: str,
        emotion: str,
        recent_context: dict[str, Any] | None,
        user_pref: str = "",
        active_personality_id: str = "default",
        personality_locked: bool = False,
        last_switch_ts: float = 0.0,
        manual_personality: str = "",
        now_ts: float | None = None,
    ) -> PersonalityDecision:
        now = float(now_ts or time.time())
        active_id = self._normalize_personality_id(active_personality_id)
        manual = self._normalize_personality_id(manual_personality)

        if manual and manual != "auto":
            target = manual if manual in self._profiles else active_id
            switched = target != active_id
            return PersonalityDecision(
                target_personality_id=target,
                confidence=1.0,
                reason="manual",
                switched=switched,
                lock_after_switch=True,
                blend=self._build_blend(active_id, target, switched),
                ts=now,
            )

        if personality_locked and manual != "auto":
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=1.0,
                reason="locked",
                switched=False,
                lock_after_switch=True,
                blend={},
                ts=now,
            )

        scores = self._score_personalities(
            intent=str(intent or "").strip().lower(),
            emotion=str(emotion or "").strip().lower(),
            recent_context=dict(recent_context or {}),
            user_pref=self._normalize_personality_id(user_pref),
        )
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ordered:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=0.0,
                reason="no_scores",
                switched=False,
                blend={},
                ts=now,
            )

        target, best = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = best / max(0.001, best + second)
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self.min_confidence:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="low_confidence",
                switched=False,
                blend={},
                ts=now,
            )

        if target == active_id:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="already_active",
                switched=False,
                blend={},
                ts=now,
            )

        elapsed = now - float(last_switch_ts or 0.0)
        if elapsed < self.cooldown_sec:
            return PersonalityDecision(
                target_personality_id=active_id,
                confidence=confidence,
                reason="cooldown",
                switched=False,
                blend={},
                ts=now,
            )

        return PersonalityDecision(
            target_personality_id=target,
            confidence=confidence,
            reason="auto",
            switched=True,
            lock_after_switch=False,
            blend=self._build_blend(active_id, target, True),
            ts=now,
        )

    def advance_blend(self, blend: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(blend or {})
        if not bool(row.get("active")):
            return {
                "active": False,
                "from": str(row.get("from") or ""),
                "to": str(row.get("to") or ""),
                "step": int(row.get("step") or 0),
                "steps": int(row.get("steps") or 0),
                "old_weight": float(row.get("old_weight") or 0.0),
                "new_weight": float(row.get("new_weight") or 1.0),
            }

        step = int(row.get("step") or 0) + 1
        steps = max(1, int(row.get("steps") or self.blend_steps))
        frac = min(1.0, step / float(steps))
        active = step < steps
        return {
            "active": active,
            "from": str(row.get("from") or ""),
            "to": str(row.get("to") or ""),
            "step": step,
            "steps": steps,
            "old_weight": round(max(0.0, 1.0 - frac), 2),
            "new_weight": round(min(1.0, frac), 2),
        }

    def _load_profiles(self) -> dict[str, PersonalityProfile]:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        loaded: dict[str, PersonalityProfile] = {}

        for path in sorted(self.profiles_dir.glob("*.json")):
            profile = self._parse_profile_file(path)
            if profile is None:
                continue
            loaded[profile.id] = profile

        defaults = _built_in_profiles()
        for pid, profile in defaults.items():
            loaded.setdefault(pid, profile)

        if self.default_personality_id not in loaded:
            self.default_personality_id = "default"
        if self.default_personality_id not in loaded:
            loaded[self.default_personality_id] = defaults["default"]
        return loaded

    def _parse_profile_file(self, path: Path) -> PersonalityProfile | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        pid = self._normalize_personality_id(payload.get("id") or path.stem)
        if not pid:
            return None
        traits_raw = payload.get("traits") if isinstance(payload.get("traits"), dict) else {}
        traits = {}
        for key, value in traits_raw.items():
            name = str(key or "").strip().lower()
            if not name:
                continue
            try:
                traits[name] = max(0.0, min(1.0, float(value)))
            except Exception:
                continue

        triggers = payload.get("triggers") if isinstance(payload.get("triggers"), dict) else {}
        enable = [str(x).strip().lower() for x in list(triggers.get("auto_enable_intents") or []) if str(x).strip()]
        disable = [str(x).strip().lower() for x in list(triggers.get("auto_disable_intents") or []) if str(x).strip()]

        return PersonalityProfile(
            id=pid,
            name=str(payload.get("name") or pid).strip() or pid,
            version=str(payload.get("version") or "0.0.0").strip() or "0.0.0",
            system_prompt=_normalize_prompt_ref(payload.get("system_prompt"), "system/personality_default.txt"),
            style_prompt=_normalize_prompt_ref(payload.get("style_prompt"), "system/style_default.txt"),
            rules_prompt=_normalize_prompt_ref(payload.get("rules_prompt"), "system/rules_default.txt"),
            voice_style=str(payload.get("voice_style") or "neutral").strip() or "neutral",
            llm_profile=str(payload.get("llm_profile") or "BALANCED").strip().upper() or "BALANCED",
            traits=traits,
            triggers={
                "auto_enable_intents": enable,
                "auto_disable_intents": disable,
            },
        )

    def _score_personalities(
        self,
        *,
        intent: str,
        emotion: str,
        recent_context: dict[str, Any],
        user_pref: str,
    ) -> dict[str, float]:
        scores = {pid: 0.2 for pid in self._profiles.keys()}
        scores.setdefault("default", 0.25)

        mode = str(recent_context.get("mode") or "").strip().lower()
        topic = str(recent_context.get("topic") or "").strip().lower()

        if intent in {"coding_help", "coding", "task_request", "question", "ui_request", "search"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.45
        if intent in {"chat", "chitchat", "relationship", "playful", "nsfw_flirt"}:
            scores["flirty"] = scores.get("flirty", 0.2) + 0.36
        if emotion in {"frustrated", "frustrated_angry", "angry", "sad", "sad_tired", "confused"}:
            scores["supportive"] = scores.get("supportive", 0.2) + 0.5
            scores["flirty"] = scores.get("flirty", 0.2) - 0.2
        if emotion in {"positive", "positive_excited", "excited", "playful", "playful_ironic"} and intent in {"chat", "chitchat", "relationship"}:
            scores["flirty"] = scores.get("flirty", 0.2) + 0.28

        if mode in {"coding", "task"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.22
            scores["flirty"] = scores.get("flirty", 0.2) - 0.15
        if topic in {"python", "git", "ui", "llm", "security"}:
            scores["strict"] = scores.get("strict", 0.2) + 0.1

        if user_pref and user_pref in scores:
            scores[user_pref] += 0.22

        # Trigger-based nudges from profile JSON.
        for pid, profile in self._profiles.items():
            enables = {str(x).strip().lower() for x in list((profile.triggers or {}).get("auto_enable_intents") or [])}
            disables = {str(x).strip().lower() for x in list((profile.triggers or {}).get("auto_disable_intents") or [])}
            if intent and intent in enables:
                scores[pid] = scores.get(pid, 0.2) + 0.18
            if intent and intent in disables:
                scores[pid] = scores.get(pid, 0.2) - 0.18

        # Keep scores in stable range.
        out = {}
        for pid, value in scores.items():
            out[pid] = max(0.01, min(1.5, float(value)))
        return out

    def _build_blend(self, old_id: str, new_id: str, active: bool) -> dict[str, Any]:
        if not active or old_id == new_id:
            return {
                "active": False,
                "from": old_id,
                "to": new_id,
                "step": 0,
                "steps": self.blend_steps,
                "old_weight": 0.0,
                "new_weight": 1.0,
            }
        return {
            "active": True,
            "from": old_id,
            "to": new_id,
            "step": 1,
            "steps": self.blend_steps,
            "old_weight": round(max(0.0, 1.0 - 1.0 / self.blend_steps), 2),
            "new_weight": round(min(1.0, 1.0 / self.blend_steps), 2),
        }

    @staticmethod
    def _normalize_personality_id(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})


def apply_personality(text: str, profile: str = "default") -> str:
    src = str(text or "")
    pid = str(profile or "default").strip().lower()
    if not src:
        return src
    if pid == "strict":
        return src
    if pid == "flirty":
        return src
    if pid == "supportive":
        return src
    return src


def _normalize_prompt_ref(value: Any, fallback: str) -> str:
    text = str(value or "").replace("\\", "/").strip().strip("/")
    if not text:
        return str(fallback)
    return text


def _built_in_profiles() -> dict[str, PersonalityProfile]:
    return {
        "default": PersonalityProfile(
            id="default",
            name="Default",
            version="1.0.0",
            system_prompt="system/personality_default.txt",
            style_prompt="system/style_default.txt",
            rules_prompt="system/rules_default.txt",
            voice_style="neutral",
            llm_profile="BALANCED",
            traits={"sarcasm": 0.2, "flirt": 0.1, "formality": 0.5, "empathy": 0.7},
            triggers={
                "auto_enable_intents": ["chat", "question", "task_request"],
                "auto_disable_intents": [],
            },
        ),
        "flirty": PersonalityProfile(
            id="flirty",
            name="Flirty",
            version="1.2.0",
            system_prompt="system/personality_flirty.txt",
            style_prompt="system/style_flirty.txt",
            rules_prompt="system/rules_default.txt",
            voice_style="warm",
            llm_profile="FAST",
            traits={"sarcasm": 0.55, "flirt": 0.8, "formality": 0.2, "empathy": 0.7},
            triggers={
                "auto_enable_intents": ["chat", "chitchat", "relationship", "playful"],
                "auto_disable_intents": ["coding_help", "security", "task_request", "work"],
            },
        ),
        "strict": PersonalityProfile(
            id="strict",
            name="Strict",
            version="1.1.0",
            system_prompt="system/personality_strict.txt",
            style_prompt="system/style_strict.txt",
            rules_prompt="system/rules_default.txt",
            voice_style="clear",
            llm_profile="QUALITY",
            traits={"sarcasm": 0.05, "flirt": 0.0, "formality": 0.85, "empathy": 0.55},
            triggers={
                "auto_enable_intents": ["coding_help", "task_request", "search", "ui_request"],
                "auto_disable_intents": ["relationship", "playful"],
            },
        ),
        "supportive": PersonalityProfile(
            id="supportive",
            name="Supportive",
            version="1.0.0",
            system_prompt="system/personality_default.txt",
            style_prompt="system/style_supportive.txt",
            rules_prompt="system/rules_default.txt",
            voice_style="soft",
            llm_profile="BALANCED",
            traits={"sarcasm": 0.0, "flirt": 0.05, "formality": 0.35, "empathy": 0.92},
            triggers={
                "auto_enable_intents": ["complaint", "chat"],
                "auto_disable_intents": ["coding_help"],
            },
        ),
    }


def ensure_default_personality_files(profiles_dir: str | Path | None = None) -> None:
    path = Path(profiles_dir).expanduser() if profiles_dir is not None else (DATA_DIR / "personalities")
    path.mkdir(parents=True, exist_ok=True)
    defaults = _built_in_profiles()
    for pid, profile in defaults.items():
        out = path / f"{pid}.json"
        if out.exists():
            continue
        out.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_default_prompt_files() -> None:
    root = BASE_DIR / "prompts" / "system"
    root.mkdir(parents=True, exist_ok=True)
    templates = {
        "style_default.txt": "Tone: balanced and practical.\\nStyle: concise and clear.",
        "style_flirty.txt": "Tone: playful but respectful.\\nStyle: short, warm, and light.",
        "style_strict.txt": "Tone: formal and technical.\\nStyle: structured steps and precise wording.",
        "style_supportive.txt": "Tone: calm and empathetic.\\nStyle: reassuring and practical.",
        "rules_default.txt": "Rules:\\n- Follow safety.\\n- Be accurate.\\n- Keep answers useful.",
    }
    for name, text in templates.items():
        out = root / name
        if out.exists():
            continue
        out.write_text(text + "\\n", encoding="utf-8")


__all__ = [
    "PersonalityProfile",
    "PersonalityDecision",
    "PersonalityEngine",
    "DEFAULT_COOLDOWN_SEC",
    "DEFAULT_MIN_CONFIDENCE",
    "DEFAULT_BLEND_STEPS",
    "apply_personality",
    "ensure_default_personality_files",
    "ensure_default_prompt_files",
]
