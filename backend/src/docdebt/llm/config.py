"""Model specifications.

A ``ModelSpec`` is everything the pipeline needs to call one model:
provider, provider-side name, decoding settings and token budgets. Token
budgets are per prompt pattern because reasoning prompts (CoT) and hidden
reasoning tokens of some cloud models need far more room than a one-word
verdict; the research runs showed truncation when budgets were too small.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["ollama", "openai", "anthropic", "google"]


class ModelSpec(BaseModel):
    """How to call one model.

    Attributes:
        id: key used everywhere else (prompt registry, config files, reports).
        provider: which LangChain integration to use.
        model: the provider-side model name (Ollama tag or API model id).
        temperature: decoding temperature for single-sample calls.
        max_tokens: output budget per purpose; keys are prompt patterns
            (``zeroshot``, ``cot``...), ``explain`` for the citation prompt,
            and ``default`` as fallback.
        max_tokens_cap: hard ceiling when a truncated answer is retried with
            a doubled budget.
        num_ctx: Ollama context window. The Ollama default (4096 tokens) is
            too small for prompts carrying retrieved context and silently
            truncates them, so it is always set explicitly.
        base_url: Ollama server URL, or a custom OpenAI-compatible endpoint.
        api_key_env: environment variable holding the API key (cloud only).
        api_key: the API key itself, when it comes from the application's
            saved model configuration rather than the environment. Never
            serialized, never shown.
        timeout_s: per-request timeout.
        max_retries: attempts per call (errors and truncations share this budget).
        extra: extra keyword arguments passed to the LangChain constructor
            (for example ``{"reasoning_effort": "low"}`` for OpenAI reasoning models).
        usd_per_1k_input / usd_per_1k_output: optional prices for cost tracking.
    """

    id: str
    provider: Provider
    model: str
    temperature: float = 0.0
    max_tokens: dict[str, int] = Field(default_factory=lambda: {"default": 256})
    max_tokens_cap: int = 4096
    num_ctx: int = 8192
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = Field(default=None, exclude=True, repr=False)
    timeout_s: float = 600.0
    max_retries: int = 3
    extra: dict[str, object] = Field(default_factory=dict)
    usd_per_1k_input: float | None = None
    usd_per_1k_output: float | None = None

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    def budget(self, purpose: str) -> int:
        """Output token budget for a pattern or purpose, falling back to ``default``."""
        return self.max_tokens.get(purpose, self.max_tokens.get("default", 256))

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.usd_per_1k_input is None or self.usd_per_1k_output is None:
            return None
        return (input_tokens * self.usd_per_1k_input + output_tokens * self.usd_per_1k_output) / 1000
