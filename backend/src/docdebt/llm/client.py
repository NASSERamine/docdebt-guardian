"""LangChain-based LLM client with retries, truncation handling and usage tracking.

One ``LLMClient`` wraps one ``ModelSpec``. Callers only ever see
``LLMResponse`` objects, never provider exceptions or LangChain types, so
swapping Qwen for GPT is a configuration change.

Retry policy (per call, ``spec.max_retries`` attempts in total):

* provider error -> wait with linear back-off and retry;
* "temperature not supported" error (reasoning models) -> drop the
  temperature for this client and retry without consuming an attempt;
* truncated answer (the model hit its output budget) -> retry with a
  doubled budget, up to ``spec.max_tokens_cap``. If every attempt is
  truncated, the last response is returned with ``truncated=True`` so the
  caller can treat it as an abstention instead of crashing.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from docdebt.llm.config import ModelSpec

ChatModelFactory = Callable[[ModelSpec, int, float | None], BaseChatModel]

_TRUNCATION_REASONS = {"length", "max_tokens", "MAX_TOKENS"}
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a model call fails after every retry."""


@dataclass
class LLMResponse:
    """Result of one logical call (possibly several attempts)."""

    text: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    truncated: bool = False
    attempts: int = 1
    latency_s: float = 0.0
    max_tokens_used: int = 0


@dataclass
class UsageTracker:
    """Thread-safe running totals of tokens, calls and cost."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, spec: ModelSpec, response: LLMResponse) -> None:
        cost = spec.cost_usd(response.input_tokens, response.output_tokens) or 0.0
        with self._lock:
            self.calls += 1
            self.input_tokens += response.input_tokens
            self.output_tokens += response.output_tokens
            self.cost_usd += cost
            row = self.by_model.setdefault(
                spec.id, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            )
            row["calls"] += 1
            row["input_tokens"] += response.input_tokens
            row["output_tokens"] += response.output_tokens
            row["cost_usd"] += cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def default_chat_model_factory(spec: ModelSpec, max_tokens: int, temperature: float | None) -> BaseChatModel:
    """Build the LangChain chat model for a spec. Provider packages are imported lazily."""
    kwargs: dict[str, Any] = dict(spec.extra)

    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            base_url=spec.base_url or "http://localhost:11434",
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=spec.num_ctx,
            keep_alive="30m",
            client_kwargs={"timeout": spec.timeout_s},
            **kwargs,
        )

    api_key = None
    if spec.api_key_env:
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise LLMError(
                f"Environment variable '{spec.api_key_env}' is empty; it must hold the API key "
                f"for model '{spec.id}'."
            )

    if spec.provider == "openai":
        from langchain_openai import ChatOpenAI

        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        if temperature is not None:
            kwargs["temperature"] = temperature
        from pydantic import SecretStr

        return ChatOpenAI(
            model=spec.model,
            api_key=SecretStr(api_key) if api_key else None,
            max_completion_tokens=max_tokens,  # counts hidden reasoning tokens on reasoning models
            timeout=spec.timeout_s,
            max_retries=0,  # retries are handled here, uniformly for every provider
            **kwargs,
        )

    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]

        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(
            model=spec.model,
            api_key=api_key,
            max_tokens=max_tokens,
            timeout=spec.timeout_s,
            max_retries=0,
            **kwargs,
        )

    if spec.provider == "google":
        from langchain_google_genai import (  # type: ignore[import-not-found]
            ChatGoogleGenerativeAI,
        )

        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(
            model=spec.model,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
            timeout=spec.timeout_s,
            max_retries=0,
            **kwargs,
        )

    raise LLMError(f"Unsupported provider '{spec.provider}'.")


def _finish_reason(metadata: dict[str, object]) -> str | None:
    for key in ("done_reason", "finish_reason", "stop_reason"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def clean_model_text(text: str) -> str:
    """Remove reasoning blocks some local models print (``<think>...</think>``)."""
    return _THINK_BLOCK.sub("", text).strip()


class LLMClient:
    """Call one model with the retry, truncation and usage policies of this module."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        factory: ChatModelFactory | None = None,
        tracker: UsageTracker | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self.tracker = tracker or UsageTracker()
        self._factory = factory or default_chat_model_factory
        self._sleep = sleep
        self._temperature_supported = True
        self._models: dict[tuple[int, float | None], BaseChatModel] = {}

    def _model(self, max_tokens: int, temperature: float | None) -> BaseChatModel:
        key = (max_tokens, temperature)
        if key not in self._models:
            self._models[key] = self._factory(self.spec, max_tokens, temperature)
        return self._models[key]

    def invoke(
        self,
        prompt: str,
        *,
        purpose: str = "default",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send ``prompt`` and return the (cleaned) text answer.

        Args:
            prompt: the full user prompt.
            purpose: pattern or purpose used to pick the token budget.
            max_tokens: explicit budget, overriding the spec.
            temperature: explicit temperature, overriding the spec.

        Raises:
            LLMError: when every attempt failed with an error.
        """
        budget = max_tokens or self.spec.budget(purpose)
        wanted_temperature = self.spec.temperature if temperature is None else temperature
        attempts = 0
        last_error: Exception | None = None
        last_response: LLMResponse | None = None
        started = time.perf_counter()

        while attempts < self.spec.max_retries:
            attempts += 1
            temp = wanted_temperature if self._temperature_supported else None
            try:
                message = self._model(budget, temp).invoke([HumanMessage(content=prompt)])
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                error_text = str(exc).lower()
                if (
                    self._temperature_supported
                    and "temperature" in error_text
                    and (
                        "not supported" in error_text
                        or "unsupported" in error_text
                        or "does not support" in error_text
                        or "invalid" in error_text
                    )
                ):
                    self._temperature_supported = False
                    attempts -= 1  # a configuration fix, not a failed attempt
                    continue
                last_error = exc
                if attempts < self.spec.max_retries:
                    self._sleep(2.0 * attempts)
                continue

            content = message.content
            if isinstance(content, list):  # some providers return content blocks
                content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            usage = getattr(message, "usage_metadata", None) or {}
            reason = _finish_reason(getattr(message, "response_metadata", {}) or {})
            text = clean_model_text(str(content))
            truncated = reason in _TRUNCATION_REASONS
            response = LLMResponse(
                text=text,
                model_id=self.spec.id,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                finish_reason=reason,
                truncated=truncated,
                attempts=attempts,
                max_tokens_used=budget,
            )
            self.tracker.record(self.spec, response)
            last_response = response
            if not truncated:
                break
            if budget >= self.spec.max_tokens_cap:
                break
            budget = min(budget * 2, self.spec.max_tokens_cap)

        if last_response is None:
            raise LLMError(f"Model '{self.spec.id}' failed after {attempts} attempt(s): {last_error}")
        last_response.latency_s = time.perf_counter() - started
        return last_response
