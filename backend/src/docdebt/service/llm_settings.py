"""The model the application calls, as chosen on the Model page.

One configuration per scope: ``user:<github_id>`` in sign-in mode (each user
brings their own API key), ``local`` in local mode. A pull request of
``owner/repo`` is reviewed with the configuration of the first user who added
that repository and saved one; in local mode, with the single local one.

The API key is stored encrypted (:mod:`docdebt.service.secrets`) and leaves
this module only inside a :class:`~docdebt.llm.config.ModelSpec` given to the
LLM client, never in an API response or a log line.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.fernet import Fernet

from docdebt.llm.catalog import SERVICE_MODELS, get_model_spec
from docdebt.llm.client import ChatModelFactory, LLMClient, LLMError
from docdebt.llm.config import ModelSpec
from docdebt.service.secrets import SecretError, decrypt
from docdebt.service.store import ReviewStore

LOCAL_SCOPE = "local"
_TEST_PROMPT = "Reply with the single word OK."


def user_scope(github_id: int) -> str:
    return f"user:{github_id}"


def scope_for_repo(store: ReviewStore, *, auth_enabled: bool, owner: str, repo: str) -> str | None:
    """The scope whose configuration reviews ``owner/repo``, or ``None`` if nobody saved one."""
    if not auth_enabled:
        return LOCAL_SCOPE if store.llm_config(LOCAL_SCOPE) else None
    for github_id in store.repo_users(owner, repo):
        scope = user_scope(github_id)
        if store.llm_config(scope):
            return scope
    return None


def build_spec(
    model: str,
    api_key: str,
    *,
    price_in_per_m: float | None = None,
    price_out_per_m: float | None = None,
) -> ModelSpec:
    """The spec of one of the offered models, with the key and prices of the user."""
    if model not in SERVICE_MODELS:
        raise ValueError(f"Unknown model {model!r}. Offered models: {list(SERVICE_MODELS)}.")
    values = get_model_spec(model).model_dump()
    values.update(
        api_key=api_key,
        api_key_env=None,
        usd_per_1k_input=None if price_in_per_m is None else price_in_per_m / 1000,
        usd_per_1k_output=None if price_out_per_m is None else price_out_per_m / 1000,
    )
    return ModelSpec(**values)


@dataclass(frozen=True)
class ResolvedModel:
    scope: str
    spec: ModelSpec


def resolve_model(cipher: Fernet, store: ReviewStore, scope: str | None) -> ResolvedModel | None:
    """The spec to call for ``scope``, or ``None`` (nothing saved, or an
    unreadable key, which is then reported on the status banner)."""
    if scope is None:
        return None
    row = store.llm_config(scope)
    if row is None:
        return None
    try:
        key = decrypt(cipher, row["api_key_enc"])
    except SecretError as exc:
        store.record_llm_failure(scope, str(exc))
        return None
    spec = build_spec(
        row["model"], key, price_in_per_m=row["price_in_per_m"], price_out_per_m=row["price_out_per_m"]
    )
    return ResolvedModel(scope, spec)


def test_connection(spec: ModelSpec, factory: ChatModelFactory | None = None) -> tuple[bool, str | None]:
    """One minimal call with ``spec``: ``(True, None)`` when the provider
    answers (a truncated answer still proves the key works), else ``(False,
    the provider's message)``."""
    test_spec = spec.model_copy(update={"max_retries": 1})
    test_spec.api_key = spec.api_key
    try:
        LLMClient(test_spec, factory=factory, sleep=lambda s: None).invoke(_TEST_PROMPT, max_tokens=64)
    except LLMError as exc:
        message = str(exc)
        if "attempt(s): " in message:
            message = message.split("attempt(s): ", 1)[1]
        return False, message[:500] or "The provider did not answer."
    return True, None


def public_config(row: dict | None) -> dict | None:
    """What the browser may see of a saved configuration (no key, only its hint)."""
    if row is None:
        return None
    return {
        "model": row["model"],
        "provider": get_model_spec(row["model"]).provider if row["model"] in SERVICE_MODELS else None,
        "key_hint": row["key_hint"],
        "price_in_per_m": row["price_in_per_m"],
        "price_out_per_m": row["price_out_per_m"],
        "last_test_ok": None if row["last_test_ok"] is None else bool(row["last_test_ok"]),
        "last_test_error": row["last_test_error"],
        "last_test_at": row["last_test_at"],
        "updated_at": row["updated_at"],
    }
