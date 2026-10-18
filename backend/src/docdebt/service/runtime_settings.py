"""Settings the application can change while the service runs.

Everything that decides *how* a Javadoc is analysed is fixed, for
determinism: one model (chosen on the Model page, see
:mod:`docdebt.service.llm_settings`), its fixed prompts at temperature 0,
retrieval always on, Replacement first and Regeneration on demand. What
remains here is a budget: the optional cap on the total LLM cost.

Values saved from the Settings page are stored in
:class:`~docdebt.service.store.ReviewStore` and read again for every
webhook, so a change applies to the next pull request without a restart.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from docdebt.service.config import ServiceSettings
from docdebt.service.store import ReviewStore


class RuntimeSettings(BaseModel):
    """What one review run uses.

    Attributes:
        max_total_cost_usd: once the LLM cost recorded across every
            repository reaches this, new pull requests are not reviewed;
            ``None`` (the default) means no cap.
    """

    max_total_cost_usd: float | None = Field(None, ge=0.0)


def default_runtime_settings(settings: ServiceSettings) -> RuntimeSettings:
    """The values before any override saved from the application."""
    return RuntimeSettings(max_total_cost_usd=None)


def effective_runtime_settings(settings: ServiceSettings, store: ReviewStore) -> RuntimeSettings:
    """Defaults with the saved overrides applied; a saved value that no longer
    validates is ignored rather than breaking every review."""
    values = default_runtime_settings(settings).model_dump()
    for key, value in store.get_settings().items():
        if key not in values:
            continue
        try:
            values[key] = getattr(RuntimeSettings(**{key: value}), key)
        except ValueError:
            continue
    return RuntimeSettings(**values)
