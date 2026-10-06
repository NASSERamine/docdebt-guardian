"""Settings the dashboard can change while the service runs.

The environment (``.env``, see :mod:`docdebt.service.config`) gives the
defaults; values saved from the dashboard's Settings page (stored in
:class:`~docdebt.service.store.ReviewStore`) override them. Both are read
again for every webhook, so a change applies to the next pull request
without restarting the process.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from docdebt.repair.registry import LOCKED_REPAIR_WORDINGS
from docdebt.service.config import ServiceSettings
from docdebt.service.store import ReviewStore
from docdebt.settings import JudgeSettings, RepairSettings, RouterSettings, Settings

# Only models the research screened a repair wording for can repair without a
# hand-picked wording, and only patterns that need no few-shot file: the
# service has no .docdebt.yml to provide one.
MODEL_CHOICES: tuple[str, ...] = tuple(LOCKED_REPAIR_WORDINGS)
PATTERN_CHOICES: tuple[str, ...] = ("zeroshot", "cot")


class RuntimeSettings(BaseModel):
    """What one review run uses.

    Attributes:
        repair_model: model that writes the repairs (and judges, when
            semantic detection is on).
        repair_pattern: repair prompt pattern.
        structural_only: deterministic checks only, no LLM detection call.
        confidence_threshold: below this confidence the Judge abstains
            instead of reporting semantic debt (unused when ``structural_only``).
        max_total_cost_usd: once the LLM cost recorded across every
            repository reaches this, new pull requests are not reviewed;
            ``None`` means no cap. Local Ollama models cost nothing.
    """

    repair_model: str
    repair_pattern: str
    structural_only: bool
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    max_total_cost_usd: float | None = Field(None, ge=0.0)

    @field_validator("repair_model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        if value not in MODEL_CHOICES:
            raise ValueError(f"must be one of {list(MODEL_CHOICES)}")
        return value

    @field_validator("repair_pattern")
    @classmethod
    def _known_pattern(cls, value: str) -> str:
        if value not in PATTERN_CHOICES:
            raise ValueError(f"must be one of {list(PATTERN_CHOICES)}")
        return value


def default_runtime_settings(settings: ServiceSettings) -> RuntimeSettings:
    """The values given by the environment, before any dashboard override.

    Built without validation: an environment may name a model the dashboard
    does not offer (it still works when its wording is configured elsewhere).
    """
    return RuntimeSettings.model_construct(
        repair_model=settings.repair_model,
        repair_pattern=settings.repair_pattern,
        structural_only=settings.structural_only,
        confidence_threshold=0.0,
        max_total_cost_usd=None,
    )


def effective_runtime_settings(settings: ServiceSettings, store: ReviewStore) -> RuntimeSettings:
    """Environment defaults with the dashboard's saved overrides applied.

    A saved value that no longer validates (e.g. a model removed from the
    choices) is ignored rather than breaking every review.
    """
    values = default_runtime_settings(settings).model_dump()
    for key, value in store.get_settings().items():
        if key not in values:
            continue
        # Validate the one field against a known-valid baseline, so an
        # unusual environment default cannot make every override invalid.
        try:
            values[key] = getattr(RuntimeSettings(**{**_VALID_BASELINE, key: value}), key)
        except ValueError:
            continue
    return RuntimeSettings.model_construct(**values)


_VALID_BASELINE = {"repair_model": MODEL_CHOICES[0], "repair_pattern": PATTERN_CHOICES[0], "structural_only": True}


def to_docdebt_settings(runtime: RuntimeSettings) -> Settings:
    """The pipeline configuration for one run.

    The router's cascade always names the repair model; whether it is
    actually consulted for semantic detection is decided by the graph mode
    (structural mode never calls it), not by this list.
    """
    return Settings(
        router=RouterSettings(cascade=[runtime.repair_model]),
        repair=RepairSettings(model=runtime.repair_model, pattern=runtime.repair_pattern),
        judges={runtime.repair_model: JudgeSettings(confidence_threshold=runtime.confidence_threshold)},
    )
