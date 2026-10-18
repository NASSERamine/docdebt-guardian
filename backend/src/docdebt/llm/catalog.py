"""Built-in model specs, taken from the research setup.

``SERVICE_MODELS`` is the closed list the application offers (API models of
the research, ``gpt-5.6-luna`` first, the default). The local Ollama models
stay in ``BUILTIN_MODELS`` only for the research scripts (parity, benchmark):
the application itself never uses a local model.

Token budgets reproduce the research runs (repair: 900 to 1400 tokens in the research, more for
GPT-5.6 whose hidden reasoning tokens count against the limit): local models answer a verdict
in 60 tokens but need a much larger budget for chain-of-thought; GPT-5.6
spends hidden reasoning tokens before answering, hence its larger budget.
Prices are left empty on purpose: fill them in the config file with the
current price list if you want cost tracking.

Only models actually used in the research are listed. Add others through
``.docdebt.yml`` rather than guessing their names here.
"""

from __future__ import annotations

from docdebt.llm.config import ModelSpec

BUILTIN_MODELS: dict[str, ModelSpec] = {
    "qwen2.5-coder:7b": ModelSpec(
        id="qwen2.5-coder:7b",
        provider="ollama",
        model="qwen2.5-coder:7b",
        max_tokens={"default": 60, "cot": 1200, "explain": 600, "repair": 1400},
    ),
    "codegemma:7b": ModelSpec(
        id="codegemma:7b",
        provider="ollama",
        model="codegemma:7b",
        max_tokens={"default": 60, "cot": 1200, "explain": 600, "repair": 1400},
    ),
    "deepseek-coder:6.7b-instruct": ModelSpec(
        id="deepseek-coder:6.7b-instruct",
        provider="ollama",
        model="deepseek-coder:6.7b-instruct",
        max_tokens={"default": 60, "cot": 1200, "explain": 600, "repair": 1400},
    ),
    "gpt-5.6": ModelSpec(
        id="gpt-5.6",
        provider="openai",
        model="gpt-5.6",
        api_key_env="OPENAI_API_KEY",
        max_tokens={"default": 500, "cot": 600, "explain": 1500, "repair": 2500},
        max_tokens_cap=8192,
    ),
    # GPT-5.6 Luna: same family and budgets as GPT-5.6 (hidden reasoning tokens
    # count against the output limit). Prices are set by the service from
    # DOCDEBT_CLOUD_PRICE_IN_PER_M / DOCDEBT_CLOUD_PRICE_OUT_PER_M.
    "gpt-5.6-luna": ModelSpec(
        id="gpt-5.6-luna",
        provider="openai",
        model="gpt-5.6-luna",
        api_key_env="OPENAI_API_KEY",
        max_tokens={"default": 500, "cot": 600, "explain": 1500, "repair": 2500},
        max_tokens_cap=8192,
    ),
    "claude-sonnet-5": ModelSpec(
        id="claude-sonnet-5",
        provider="anthropic",
        model="claude-sonnet-5",
        max_tokens={"default": 500, "cot": 1200, "explain": 1500, "repair": 2500},
        max_tokens_cap=8192,
    ),
    "gemini-3.1-pro": ModelSpec(
        id="gemini-3.1-pro",
        provider="google",
        model="gemini-3.1-pro",
        max_tokens={"default": 1000, "cot": 2000, "explain": 2000, "repair": 4000},
        max_tokens_cap=16384,
    ),
}

# The models the application offers, in display order; the first is the default.
SERVICE_MODELS: tuple[str, ...] = ("gpt-5.6-luna", "gpt-5.6", "claude-sonnet-5", "gemini-3.1-pro")
DEFAULT_SERVICE_MODEL = SERVICE_MODELS[0]
PROVIDER_LABELS = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google"}
MODEL_LABELS = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6": "GPT-5.6",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
}


def service_model_choices() -> list[dict]:
    """The models the application offers, grouped by provider, for the Model page."""
    return [
        {
            "id": model_id,
            "label": MODEL_LABELS[model_id],
            "provider": BUILTIN_MODELS[model_id].provider,
            "provider_label": PROVIDER_LABELS[BUILTIN_MODELS[model_id].provider],
            "default": model_id == DEFAULT_SERVICE_MODEL,
        }
        for model_id in SERVICE_MODELS
    ]


def get_model_spec(model_id: str, overrides: dict[str, ModelSpec] | None = None) -> ModelSpec:
    """Spec for ``model_id``: user overrides first, then the built-in catalog."""
    if overrides and model_id in overrides:
        return overrides[model_id]
    if model_id in BUILTIN_MODELS:
        return BUILTIN_MODELS[model_id]
    known = sorted({*BUILTIN_MODELS, *(overrides or {})})
    raise KeyError(f"Unknown model '{model_id}'. Known models: {known}. Declare it in .docdebt.yml.")
