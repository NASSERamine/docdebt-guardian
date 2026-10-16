"""Settings the dashboard can change while the service runs.

The environment (``.env``, see :mod:`docdebt.service.config`) gives the
defaults; values saved from the dashboard's Settings page (stored in
:class:`~docdebt.service.store.ReviewStore`) override them. Both are read
again for every webhook, so a change applies to the next pull request
without restarting the process.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from docdebt.rag.retriever import DEFAULT_SOURCES, SOURCES
from docdebt.repair.registry import LOCKED_REPAIR_WORDINGS
from docdebt.service.config import ServiceSettings
from docdebt.service.store import ReviewStore
from docdebt.settings import JudgeSettings, RepairSettings, RouterSettings, Settings

# Only models the research screened a repair wording for can repair without a
# hand-picked wording, and only patterns that need no few-shot file: the
# service has no .docdebt.yml to provide one.
MODEL_CHOICES: tuple[str, ...] = tuple(LOCKED_REPAIR_WORDINGS)
PATTERN_CHOICES: tuple[str, ...] = ("zeroshot", "cot")
MECHANISM_CHOICES: tuple[str, ...] = ("hybrid", "regeneration")


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
        repair_mechanism: ``hybrid`` (research hybrid Replacement) or
            ``regeneration`` (whole-block regeneration, screened wordings).
        rag_repair / rag_detection: give the retrieved project context to
            the repair prompt / the detection prompts.
        rag_top_k: similar project methods added to the context.
        rag_sources: context sources (see ``docdebt.rag.retriever.SOURCES``).
    """

    repair_model: str
    repair_pattern: str
    structural_only: bool
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    max_total_cost_usd: float | None = Field(None, ge=0.0)
    repair_mechanism: Literal["hybrid", "regeneration"] = "hybrid"
    rag_repair: bool = True
    rag_detection: bool = False
    rag_top_k: int = Field(2, ge=1, le=5)
    rag_sources: list[str] = Field(default_factory=lambda: list(DEFAULT_SOURCES))

    @field_validator("rag_sources")
    @classmethod
    def _known_sources(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(SOURCES))
        if unknown:
            raise ValueError(f"unknown sources {unknown}; valid: {list(SOURCES)}")
        return [s for s in SOURCES if s in value]

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
        repair_mechanism=settings.repair_mechanism if settings.repair_mechanism in MECHANISM_CHOICES else "hybrid",
        rag_repair=True,
        rag_detection=False,
        rag_top_k=2,
        rag_sources=list(DEFAULT_SOURCES),
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


def to_docdebt_settings(
    runtime: RuntimeSettings,
    *,
    cloud_model: str | None = None,
    allow_cloud: bool = False,
    confidence_threshold: float | None = None,
    models: dict[str, dict] | None = None,
) -> Settings:
    """The pipeline configuration for one run.

    The cascade always starts with the local model; the cloud model is added
    only when one is configured and the repository allows it. Whether the
    cascade is consulted for semantic detection at all is decided by the
    graph mode (structural mode never calls it), not by this list.
    ``confidence_threshold`` (a per-repository value) overrides the global one.
    """
    cascade = [runtime.repair_model]
    if cloud_model and allow_cloud and cloud_model != runtime.repair_model:
        cascade.append(cloud_model)
    threshold = runtime.confidence_threshold if confidence_threshold is None else confidence_threshold
    return Settings(
        models=models or {},
        router=RouterSettings(cascade=cascade, allow_cloud=allow_cloud),
        repair=RepairSettings(model=runtime.repair_model, pattern=runtime.repair_pattern),
        judges={
            model: JudgeSettings(confidence_threshold=threshold, use_context=runtime.rag_detection)
            for model in cascade
        },
    )
