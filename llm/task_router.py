from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from config.settings import load_config
from llm.provider_base import LLMProviderBase, LLMRequest, LLMResponse, Message
from llm.task_models import TaskModelProfile, TaskModelRegistry, get_task_model_registry
from llm.task_structured import (
    TaskOutputSpec,
    TaskOutputValidationError,
    normalize_output_spec,
    validate_task_output,
)
from utils.logger import get_logger, log_json


LOGGER = get_logger(__name__)


class TaskModelExecutionError(RuntimeError):
    pass


class TaskModelNotFoundError(TaskModelExecutionError):
    pass


class TaskModelDisabledError(TaskModelExecutionError):
    pass


class TaskModelValidationError(TaskModelExecutionError):
    pass


ProviderFactory = Callable[[TaskModelProfile], LLMProviderBase]


@dataclass(frozen=True)
class TaskModelAttempt:
    task_name: str
    profile_name: str
    provider: str
    model: str
    retry_index: int
    fallback_used: bool
    duration_ms: float
    success: bool
    validation_passed: bool
    status: str
    error: str = ""
    parse_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "profile_name": self.profile_name,
            "provider": self.provider,
            "model": self.model,
            "retry_index": self.retry_index,
            "fallback_used": self.fallback_used,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "validation_passed": self.validation_passed,
            "status": self.status,
            "error": self.error,
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class TaskModelFailureContext:
    task_name: str
    json_mode: bool
    output_spec: TaskOutputSpec
    attempted_profiles: tuple[str, ...]
    attempts: tuple[TaskModelAttempt, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "json_mode": self.json_mode,
            "output_spec": {
                "kind": self.output_spec.kind,
                "required_fields": list(self.output_spec.required_fields),
                "allowed_labels": list(self.output_spec.allowed_labels),
                "max_chars": self.output_spec.max_chars,
            },
            "attempted_profiles": list(self.attempted_profiles),
            "attempts": [row.to_dict() for row in self.attempts],
            "errors": list(self.errors),
        }


DeterministicFallback = Callable[[TaskModelFailureContext], Any]


@dataclass(frozen=True)
class TaskModelExecutionResult:
    task_name: str
    profile_name: str
    provider: str
    model: str
    text: str
    response: LLMResponse
    used_fallback: bool = False
    attempted_profiles: tuple[str, ...] = ()
    json_payload: Any = None
    parsed_output: Any = None
    validation_passed: bool = True
    validation_errors: tuple[str, ...] = ()
    fallback_reason: str = ""
    deterministic_fallback_used: bool = False
    retries_used: int = 0
    latency_ms: float = 0.0
    attempts: tuple[TaskModelAttempt, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "profile_name": self.profile_name,
            "provider": self.provider,
            "model": self.model,
            "text": self.text,
            "used_fallback": self.used_fallback,
            "attempted_profiles": list(self.attempted_profiles),
            "json_payload": self.json_payload,
            "parsed_output": self.parsed_output,
            "validation_passed": self.validation_passed,
            "validation_errors": list(self.validation_errors),
            "fallback_reason": self.fallback_reason,
            "deterministic_fallback_used": self.deterministic_fallback_used,
            "retries_used": self.retries_used,
            "latency_ms": self.latency_ms,
            "attempts": [row.to_dict() for row in self.attempts],
        }


def _normalize_name(value: str | None) -> str:
    return str(value or "").strip().lower()


def _default_provider_factory(profile: TaskModelProfile) -> LLMProviderBase:
    provider_name = _normalize_name(profile.provider)
    if provider_name in {"", "auto"}:
        provider_name = _normalize_name(getattr(load_config(), "llm_default_provider", "ollama"))
    if provider_name in {"", "auto"}:
        provider_name = "ollama"
    if provider_name == "openai":
        from llm.openai_provider import OpenAIProvider

        return OpenAIProvider(default_model=profile.model, timeout_sec=profile.timeout)
    from llm.ollama_provider import OllamaProvider

    return OllamaProvider(default_model=profile.model, timeout_sec=profile.timeout)


def _synthetic_response(text: str, *, model: str = "deterministic_fallback") -> LLMResponse:
    return LLMResponse(text=str(text or ""), model=str(model or "deterministic_fallback"))


def _coerce_output_spec(
    *,
    json_mode: bool,
    output_spec: TaskOutputSpec | None,
    required_fields: list[str] | tuple[str, ...] | None,
    max_output_chars: int | None,
    allowed_labels: list[str] | tuple[str, ...] | None,
    label_aliases: dict[str, str] | None,
    allow_array: bool,
) -> TaskOutputSpec:
    kind = "json" if json_mode else ("classification" if allowed_labels else "text")
    return normalize_output_spec(
        output_spec,
        kind=kind,
        max_chars=(int(max_output_chars) if max_output_chars is not None else (16000 if json_mode else 4000)),
        required_fields=required_fields,
        allowed_labels=allowed_labels,
        label_aliases=label_aliases,
        allow_array=allow_array,
    )


def _count_retries(attempts: list[TaskModelAttempt]) -> int:
    return sum(1 for attempt in attempts if int(attempt.retry_index) > 0)


def _summarize_errors(errors: list[Exception]) -> list[str]:
    out: list[str] = []
    for exc in errors:
        text = str(exc or "").strip() or type(exc).__name__
        if text not in out:
            out.append(text)
    return out


class TaskModelRouter:
    def __init__(
        self,
        *,
        registry: TaskModelRegistry | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._registry = registry or get_task_model_registry()
        self._provider_factory = provider_factory or _default_provider_factory

    def get_task_profile(self, task_name: str) -> TaskModelProfile | None:
        return self._registry.get_task_profile(task_name)

    def is_task_enabled(self, task_name: str) -> bool:
        return self._registry.is_task_enabled(task_name)

    def run_task_model(
        self,
        task_name: str,
        prompt: str,
        *,
        system_prompt: str = "",
        metadata: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        allow_fallback: bool = True,
        deterministic_fallback: DeterministicFallback | None = None,
        output_spec: TaskOutputSpec | None = None,
        max_output_chars: int | None = None,
        max_retries: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> TaskModelExecutionResult:
        return self._run(
            task_name=task_name,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata=metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=False,
            response_format=None,
            allow_fallback=allow_fallback,
            deterministic_fallback=deterministic_fallback,
            output_spec=_coerce_output_spec(
                json_mode=False,
                output_spec=output_spec,
                required_fields=None,
                max_output_chars=max_output_chars,
                allowed_labels=None,
                label_aliases=None,
                allow_array=False,
            ),
            max_retries=max_retries,
            context=context,
        )

    def run_task_model_json(
        self,
        task_name: str,
        prompt: str,
        *,
        system_prompt: str = "",
        metadata: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
        required_fields: list[str] | tuple[str, ...] | None = None,
        allow_array: bool = False,
        allow_fallback: bool = True,
        deterministic_fallback: DeterministicFallback | None = None,
        output_spec: TaskOutputSpec | None = None,
        max_output_chars: int | None = None,
        max_retries: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> TaskModelExecutionResult:
        json_system_prompt = "Return valid JSON only."
        effective_system_prompt = json_system_prompt
        if str(system_prompt or "").strip():
            effective_system_prompt = f"{str(system_prompt).strip()}\n\n{json_system_prompt}"
        return self._run(
            task_name=task_name,
            prompt=prompt,
            system_prompt=effective_system_prompt,
            metadata=metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=True,
            response_format=dict(response_format) if isinstance(response_format, dict) else None,
            allow_fallback=allow_fallback,
            deterministic_fallback=deterministic_fallback,
            output_spec=_coerce_output_spec(
                json_mode=True,
                output_spec=output_spec,
                required_fields=required_fields,
                max_output_chars=max_output_chars,
                allowed_labels=None,
                label_aliases=None,
                allow_array=allow_array,
            ),
            max_retries=max_retries,
            context=context,
        )

    def run_task_model_choice(
        self,
        task_name: str,
        prompt: str,
        *,
        system_prompt: str = "",
        metadata: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        allowed_labels: list[str] | tuple[str, ...] | None = None,
        label_aliases: dict[str, str] | None = None,
        allow_fallback: bool = True,
        deterministic_fallback: DeterministicFallback | None = None,
        output_spec: TaskOutputSpec | None = None,
        max_output_chars: int | None = None,
        max_retries: int = 0,
        context: Mapping[str, Any] | None = None,
    ) -> TaskModelExecutionResult:
        return self._run(
            task_name=task_name,
            prompt=prompt,
            system_prompt=system_prompt,
            metadata=metadata,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            json_mode=False,
            response_format=None,
            allow_fallback=allow_fallback,
            deterministic_fallback=deterministic_fallback,
            output_spec=_coerce_output_spec(
                json_mode=False,
                output_spec=output_spec,
                required_fields=None,
                max_output_chars=max_output_chars or 128,
                allowed_labels=allowed_labels,
                label_aliases=label_aliases,
                allow_array=False,
            ),
            max_retries=max_retries,
            context=context,
        )

    def _build_profile_chain(self, task_name: str, *, allow_fallback: bool) -> list[TaskModelProfile]:
        name = _normalize_name(task_name)
        primary = self._registry.get_task_profile(name)
        if primary is None:
            raise TaskModelNotFoundError(f"Task model profile not found: {name}")
        chain: list[TaskModelProfile] = [primary]
        if not allow_fallback:
            return chain
        seen = {primary.name}
        current = primary
        while current.fallback_profile:
            next_profile = self._registry.get_task_profile(current.fallback_profile)
            if next_profile is None or next_profile.name in seen:
                break
            chain.append(next_profile)
            seen.add(next_profile.name)
            current = next_profile
        return chain

    def _run(
        self,
        *,
        task_name: str,
        prompt: str,
        system_prompt: str,
        metadata: Mapping[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        timeout: float | None,
        json_mode: bool,
        response_format: dict[str, Any] | None,
        allow_fallback: bool,
        deterministic_fallback: DeterministicFallback | None,
        output_spec: TaskOutputSpec,
        max_retries: int,
        context: Mapping[str, Any] | None,
    ) -> TaskModelExecutionResult:
        normalized_task = _normalize_name(task_name)
        chain = self._build_profile_chain(normalized_task, allow_fallback=allow_fallback)
        started_at = time.perf_counter()
        log_json(
            LOGGER,
            "task_model_run_start",
            context=context,
            task_name=normalized_task,
            json_mode=bool(json_mode),
            profiles=[profile.name for profile in chain],
            fallback_allowed=bool(allow_fallback),
            max_retries=max(0, int(max_retries)),
            output_kind=str(output_spec.kind),
            required_fields=list(output_spec.required_fields),
            allowed_labels=list(output_spec.allowed_labels),
        )

        attempts: list[TaskModelAttempt] = []
        errors: list[Exception] = []
        attempted_profiles: list[str] = []
        retries_limit = max(0, int(max_retries))

        for idx, base_profile in enumerate(chain):
            profile = replace(
                base_profile,
                temperature=float(temperature) if temperature is not None else float(base_profile.temperature),
                max_tokens=int(max_tokens) if max_tokens is not None else int(base_profile.max_tokens),
                timeout=float(timeout) if timeout is not None else float(base_profile.timeout),
            )
            attempted_profiles.append(profile.name)
            fallback_used = idx > 0

            if not profile.enabled:
                error = TaskModelDisabledError(f"Task model profile is disabled: {profile.name}")
                errors.append(error)
                attempt = TaskModelAttempt(
                    task_name=normalized_task,
                    profile_name=profile.name,
                    provider=profile.provider,
                    model=profile.model,
                    retry_index=0,
                    fallback_used=fallback_used,
                    duration_ms=0.0,
                    success=False,
                    validation_passed=False,
                    status="disabled",
                    error=str(error),
                )
                attempts.append(attempt)
                log_json(
                    LOGGER,
                    "task_model_run_skipped",
                    context=context,
                    task_name=normalized_task,
                    profile_name=profile.name,
                    provider=profile.provider,
                    model=profile.model,
                    used_fallback=fallback_used,
                    retry_index=0,
                    validation_passed=False,
                    reason="profile_disabled",
                )
                continue

            for retry_index in range(retries_limit + 1):
                attempt_started = time.perf_counter()
                log_json(
                    LOGGER,
                    "task_model_run_attempt",
                    context=context,
                    task_name=normalized_task,
                    profile_name=profile.name,
                    provider=profile.provider,
                    model=profile.model,
                    used_fallback=fallback_used,
                    retry_index=retry_index,
                    json_mode=bool(json_mode),
                )
                try:
                    provider = self._provider_factory(profile)
                    request = self._build_request(
                        profile=profile,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        metadata=metadata,
                        json_mode=json_mode,
                        response_format=response_format,
                    )
                    response = provider.generate(request)
                    clean_text, parsed_output = self._validate_response(
                        text=str(response.text or ""),
                        output_spec=output_spec,
                    )
                    duration_ms = (time.perf_counter() - attempt_started) * 1000.0
                    success_attempt = TaskModelAttempt(
                        task_name=normalized_task,
                        profile_name=profile.name,
                        provider=profile.provider,
                        model=str(response.model or profile.model),
                        retry_index=retry_index,
                        fallback_used=fallback_used,
                        duration_ms=round(float(duration_ms), 2),
                        success=True,
                        validation_passed=True,
                        status="ok",
                    )
                    attempts.append(success_attempt)
                    total_latency_ms = (time.perf_counter() - started_at) * 1000.0
                    json_payload = parsed_output if output_spec.kind == "json" else None
                    log_json(
                        LOGGER,
                        "task_model_run_done",
                        context=context,
                        summary=(
                            f"task={normalized_task} profile={profile.name} "
                            f"provider={profile.provider} success=true fallback={str(fallback_used).lower()}"
                        ),
                        task_name=normalized_task,
                        profile_name=profile.name,
                        provider=profile.provider,
                        model=str(response.model or profile.model),
                        used_fallback=fallback_used,
                        success=True,
                        json_mode=bool(json_mode),
                        retry_index=retry_index,
                        retries_used=_count_retries(attempts),
                        duration_ms=round(float(duration_ms), 2),
                        total_latency_ms=round(float(total_latency_ms), 2),
                        validation_passed=True,
                    )
                    return TaskModelExecutionResult(
                        task_name=normalized_task,
                        profile_name=profile.name,
                        provider=profile.provider,
                        model=str(response.model or profile.model),
                        text=clean_text,
                        response=response,
                        used_fallback=fallback_used,
                        attempted_profiles=tuple(attempted_profiles),
                        json_payload=json_payload,
                        parsed_output=parsed_output,
                        validation_passed=True,
                        validation_errors=(),
                        fallback_reason=("profile_fallback" if fallback_used else ""),
                        deterministic_fallback_used=False,
                        retries_used=_count_retries(attempts),
                        latency_ms=round(float(total_latency_ms), 2),
                        attempts=tuple(attempts),
                    )
                except Exception as exc:
                    duration_ms = (time.perf_counter() - attempt_started) * 1000.0
                    wrapped_exc, validation_passed, parse_error = self._normalize_error(exc)
                    errors.append(wrapped_exc)
                    failure_attempt = TaskModelAttempt(
                        task_name=normalized_task,
                        profile_name=profile.name,
                        provider=profile.provider,
                        model=profile.model,
                        retry_index=retry_index,
                        fallback_used=fallback_used,
                        duration_ms=round(float(duration_ms), 2),
                        success=False,
                        validation_passed=validation_passed,
                        status=("validation_failed" if isinstance(wrapped_exc, TaskModelValidationError) else "failed"),
                        error=str(wrapped_exc),
                        parse_error=parse_error,
                    )
                    attempts.append(failure_attempt)
                    log_json(
                        LOGGER,
                        "task_model_run_failed",
                        context=context,
                        summary=(
                            f"task={normalized_task} profile={profile.name} "
                            f"provider={profile.provider} success=false fallback={str(fallback_used).lower()}"
                        ),
                        task_name=normalized_task,
                        profile_name=profile.name,
                        provider=profile.provider,
                        model=profile.model,
                        used_fallback=fallback_used,
                        success=False,
                        json_mode=bool(json_mode),
                        retry_index=retry_index,
                        retries_used=_count_retries(attempts),
                        duration_ms=round(float(duration_ms), 2),
                        validation_passed=validation_passed,
                        parse_error=parse_error,
                        error=str(wrapped_exc),
                    )
                    if retry_index < retries_limit:
                        continue
                    break

        failure_context = TaskModelFailureContext(
            task_name=normalized_task,
            json_mode=bool(json_mode),
            output_spec=output_spec,
            attempted_profiles=tuple(attempted_profiles),
            attempts=tuple(attempts),
            errors=tuple(_summarize_errors(errors)),
        )
        deterministic_result = self._try_deterministic_fallback(
            deterministic_fallback=deterministic_fallback,
            failure=failure_context,
            context=context,
            started_at=started_at,
        )
        if deterministic_result is not None:
            return deterministic_result

        log_json(
            LOGGER,
            "task_model_run_exhausted",
            context=context,
            summary=f"task={normalized_task} exhausted=true attempts={len(attempts)}",
            task_name=normalized_task,
            profiles=list(attempted_profiles),
            attempts=[row.to_dict() for row in attempts],
            errors=list(failure_context.errors),
            retries_used=_count_retries(attempts),
            total_latency_ms=round(float((time.perf_counter() - started_at) * 1000.0), 2),
        )
        if len(errors) == 1 and isinstance(errors[-1], TaskModelExecutionError):
            raise errors[-1]
        last_error = errors[-1] if errors else None
        message = (
            f"Task model execution failed: {normalized_task}; "
            f"attempted_profiles={list(attempted_profiles)}; errors={list(failure_context.errors)}"
        )
        raise TaskModelExecutionError(message) from last_error

    def _try_deterministic_fallback(
        self,
        *,
        deterministic_fallback: DeterministicFallback | None,
        failure: TaskModelFailureContext,
        context: Mapping[str, Any] | None,
        started_at: float,
    ) -> TaskModelExecutionResult | None:
        if deterministic_fallback is None:
            return None
        try:
            value = deterministic_fallback(failure)
        except Exception as exc:
            log_json(
                LOGGER,
                "task_model_deterministic_fallback_failed",
                context=context,
                task_name=failure.task_name,
                error=str(exc),
            )
            return None
        if value is None:
            return None
        if isinstance(value, TaskModelExecutionResult):
            return value
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value or "")
        try:
            clean_text, parsed_output = self._validate_response(text=text, output_spec=failure.output_spec)
        except Exception as exc:
            log_json(
                LOGGER,
                "task_model_deterministic_fallback_failed",
                context=context,
                task_name=failure.task_name,
                error=str(exc),
            )
            return None
        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        attempt = TaskModelAttempt(
            task_name=failure.task_name,
            profile_name="deterministic_fallback",
            provider="deterministic",
            model="deterministic_fallback",
            retry_index=0,
            fallback_used=True,
            duration_ms=0.0,
            success=True,
            validation_passed=True,
            status="deterministic_fallback",
        )
        attempts = list(failure.attempts) + [attempt]
        json_payload = parsed_output if failure.output_spec.kind == "json" else None
        log_json(
            LOGGER,
            "task_model_deterministic_fallback_used",
            context=context,
            summary=f"task={failure.task_name} deterministic_fallback=true",
            task_name=failure.task_name,
            provider="deterministic",
            model="deterministic_fallback",
            used_fallback=True,
            retries_used=_count_retries(attempts),
            validation_passed=True,
            total_latency_ms=round(float(total_latency_ms), 2),
        )
        return TaskModelExecutionResult(
            task_name=failure.task_name,
            profile_name="deterministic_fallback",
            provider="deterministic",
            model="deterministic_fallback",
            text=clean_text,
            response=_synthetic_response(clean_text),
            used_fallback=True,
            attempted_profiles=tuple(failure.attempted_profiles),
            json_payload=json_payload,
            parsed_output=parsed_output,
            validation_passed=True,
            validation_errors=(),
            fallback_reason="deterministic_fallback",
            deterministic_fallback_used=True,
            retries_used=_count_retries(attempts),
            latency_ms=round(float(total_latency_ms), 2),
            attempts=tuple(attempts),
        )

    @staticmethod
    def _normalize_error(exc: Exception) -> tuple[Exception, bool, str]:
        if isinstance(exc, TaskModelValidationError):
            return exc, False, str(exc)
        if isinstance(exc, TaskOutputValidationError):
            wrapped = TaskModelValidationError(str(exc))
            return wrapped, False, str(exc)
        return exc, False, ""

    @staticmethod
    def _validate_response(*, text: str, output_spec: TaskOutputSpec) -> tuple[str, Any]:
        try:
            return validate_task_output(text, output_spec)
        except TaskOutputValidationError as exc:
            raise TaskModelValidationError(str(exc)) from exc

    @staticmethod
    def _build_request(
        *,
        profile: TaskModelProfile,
        prompt: str,
        system_prompt: str,
        metadata: Mapping[str, Any] | None,
        json_mode: bool,
        response_format: dict[str, Any] | None,
    ) -> LLMRequest:
        messages: list[Message] = []
        if str(system_prompt or "").strip():
            messages.append(Message(role="system", content=str(system_prompt).strip()))
        messages.append(Message(role="user", content=str(prompt or "")))
        req_metadata = dict(metadata or {})
        req_metadata["task_name"] = profile.name
        req_metadata["task_model_profile"] = profile.name
        return LLMRequest(
            messages=messages,
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            json_mode=bool(json_mode),
            response_format=(dict(response_format) if isinstance(response_format, dict) else None),
            metadata=req_metadata,
        )


def get_task_router(*, force_reload: bool = False) -> TaskModelRouter:
    registry = get_task_model_registry(force_reload=force_reload)
    return TaskModelRouter(registry=registry)


def run_task_model(
    task_name: str,
    prompt: str,
    *,
    system_prompt: str = "",
    metadata: Mapping[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    allow_fallback: bool = True,
    deterministic_fallback: DeterministicFallback | None = None,
    output_spec: TaskOutputSpec | None = None,
    max_output_chars: int | None = None,
    max_retries: int = 0,
    context: Mapping[str, Any] | None = None,
    force_reload: bool = False,
) -> TaskModelExecutionResult:
    return get_task_router(force_reload=force_reload).run_task_model(
        task_name=task_name,
        prompt=prompt,
        system_prompt=system_prompt,
        metadata=metadata,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        allow_fallback=allow_fallback,
        deterministic_fallback=deterministic_fallback,
        output_spec=output_spec,
        max_output_chars=max_output_chars,
        max_retries=max_retries,
        context=context,
    )


def run_task_model_json(
    task_name: str,
    prompt: str,
    *,
    system_prompt: str = "",
    metadata: Mapping[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    response_format: dict[str, Any] | None = None,
    required_fields: list[str] | tuple[str, ...] | None = None,
    allow_array: bool = False,
    allow_fallback: bool = True,
    deterministic_fallback: DeterministicFallback | None = None,
    output_spec: TaskOutputSpec | None = None,
    max_output_chars: int | None = None,
    max_retries: int = 0,
    context: Mapping[str, Any] | None = None,
    force_reload: bool = False,
) -> TaskModelExecutionResult:
    return get_task_router(force_reload=force_reload).run_task_model_json(
        task_name=task_name,
        prompt=prompt,
        system_prompt=system_prompt,
        metadata=metadata,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format=response_format,
        required_fields=required_fields,
        allow_array=allow_array,
        allow_fallback=allow_fallback,
        deterministic_fallback=deterministic_fallback,
        output_spec=output_spec,
        max_output_chars=max_output_chars,
        max_retries=max_retries,
        context=context,
    )


def run_task_model_choice(
    task_name: str,
    prompt: str,
    *,
    system_prompt: str = "",
    metadata: Mapping[str, Any] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    allowed_labels: list[str] | tuple[str, ...] | None = None,
    label_aliases: dict[str, str] | None = None,
    allow_fallback: bool = True,
    deterministic_fallback: DeterministicFallback | None = None,
    output_spec: TaskOutputSpec | None = None,
    max_output_chars: int | None = None,
    max_retries: int = 0,
    context: Mapping[str, Any] | None = None,
    force_reload: bool = False,
) -> TaskModelExecutionResult:
    return get_task_router(force_reload=force_reload).run_task_model_choice(
        task_name=task_name,
        prompt=prompt,
        system_prompt=system_prompt,
        metadata=metadata,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        allowed_labels=allowed_labels,
        label_aliases=label_aliases,
        allow_fallback=allow_fallback,
        deterministic_fallback=deterministic_fallback,
        output_spec=output_spec,
        max_output_chars=max_output_chars,
        max_retries=max_retries,
        context=context,
    )
