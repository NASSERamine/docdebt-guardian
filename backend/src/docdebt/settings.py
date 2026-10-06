"""Configuration file (``.docdebt.yml``) loading.

Every relative path in the file is resolved against the directory of the
file itself, never against the current working directory, so the same
configuration works from a terminal, a notebook or a CI job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from docdebt.core.auditor import AuditorConfig
from docdebt.core.extractor import ExtractorConfig
from docdebt.core.models import Visibility
from docdebt.detection.judge import JudgeConfig
from docdebt.detection.router import RouterPolicy
from docdebt.llm.config import ModelSpec

DEFAULT_CONFIG_NAME = ".docdebt.yml"


class ExtractorSettings(BaseModel):
    visibilities: list[Visibility] = Field(default_factory=lambda: list(Visibility))
    require_javadoc: bool = True
    skip_syntax_errors: bool = True
    exclude: list[str] = Field(default_factory=list)

    def to_config(self) -> ExtractorConfig:
        return ExtractorConfig(
            visibilities=frozenset(self.visibilities),
            require_javadoc=self.require_javadoc,
            skip_syntax_errors=self.skip_syntax_errors,
            exclude=tuple(self.exclude),
        )


class AuditorSettings(BaseModel):
    require_throws: bool = True
    hygiene_warnings: bool = True

    def to_config(self) -> AuditorConfig:
        return AuditorConfig(**self.model_dump())


class JudgeSettings(BaseModel):
    patterns: list[str] = Field(default_factory=lambda: ["zeroshot"])
    samples_per_pattern: int = 1
    sample_temperature: float = 0.7
    explain: bool = True
    require_citation: bool = True
    confidence_threshold: float = 0.0
    use_context: bool = False

    def to_config(self) -> JudgeConfig:
        return JudgeConfig(**self.model_dump())


class RouterSettings(BaseModel):
    cascade: list[str] = Field(default_factory=lambda: ["qwen2.5-coder:7b"])
    allow_cloud: bool = False
    max_tokens_total: int | None = None
    max_usd_total: float | None = None
    skip_llm_when_structural: bool = False

    def to_policy(self) -> RouterPolicy:
        return RouterPolicy(**self.model_dump())


class RepairSettings(BaseModel):
    """Repair configuration. ``model`` defaults to the first model of the router cascade."""

    model: str | None = None
    pattern: str = "zeroshot"
    max_attempts: int = 2
    hint_findings: bool = False
    fewshot_file: str | None = None
    wording_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)


class ContextSettings(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["siblings"])


class GraphSettings(BaseModel):
    """LangGraph orchestration configuration.

    Attributes:
        checkpoint_path: SQLite file the graph's state is saved to, so a
            paused review survives a process restart. Relative to the
            directory of the configuration file. Defaults to
            ``.docdebt/graph.sqlite3`` under that same directory.
    """

    checkpoint_path: str | None = None


class Settings(BaseModel):
    """Full configuration. Every section is optional."""

    extractor: ExtractorSettings = Field(default_factory=ExtractorSettings)
    auditor: AuditorSettings = Field(default_factory=AuditorSettings)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    wording_overrides: dict[str, dict[str, int]] = Field(default_factory=dict)
    fewshot_file: str | None = None
    judges: dict[str, JudgeSettings] = Field(default_factory=dict)
    router: RouterSettings = Field(default_factory=RouterSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    repair: RepairSettings = Field(default_factory=RepairSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    workers: int = 1
    base_dir: str = "."

    def model_specs(self) -> dict[str, ModelSpec]:
        """User-declared specs; the model id defaults to its key."""
        return {key: ModelSpec(**{"id": key, **raw}) for key, raw in self.models.items()}

    def judge_config(self, model_id: str) -> JudgeConfig:
        return self.judges.get(model_id, JudgeSettings()).to_config()

    def resolve(self, path: str | None) -> Path | None:
        if path is None:
            return None
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (Path(self.base_dir) / candidate).resolve()


def load_settings(path: str | Path | None = None, search_from: str | Path | None = None) -> Settings:
    """Load settings from ``path``, or from ``.docdebt.yml`` found in ``search_from``
    or its parents; defaults when no file exists.

    With no config file, ``base_dir`` still anchors on ``search_from`` (the
    file or directory being analysed) rather than the current working
    directory: every relative path in this object — including where
    ``docdebt review`` puts its checkpoint database — must stay next to the
    code being analysed, not wherever the command happened to be launched
    from, or a paused review could not be found again after a restart.
    """
    start: Path | None = None
    if search_from is not None:
        start = Path(search_from).resolve()
        start = start if start.is_dir() else start.parent
    if path is None and start is not None:
        for directory in (start, *start.parents):
            candidate = directory / DEFAULT_CONFIG_NAME
            if candidate.is_file():
                path = candidate
                break
    if path is None:
        return Settings(base_dir=str(start or Path.cwd()))
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw["base_dir"] = str(path.parent)
    return Settings(**raw)
