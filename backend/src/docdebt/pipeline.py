"""Assemble Extractor, Auditor and Router into a detection pipeline.

This is a plain sequential pipeline for the CLI and the benchmark scripts.
The LangGraph graph (phase 3) will call the same components node by node.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from docdebt.core.auditor import Auditor
from docdebt.core.models import MethodUnit
from docdebt.detection.context import ContextRetriever
from docdebt.detection.judge import Judge
from docdebt.detection.router import RoutedResult, Router
from docdebt.llm.catalog import get_model_spec
from docdebt.llm.client import ChatModelFactory, LLMClient, UsageTracker
from docdebt.llm.prompts.fewshot import FewShotSet
from docdebt.llm.prompts.registry import PromptRegistry
from docdebt.repair.fewshot import RepairFewShotSet
from docdebt.repair.registry import RepairRegistry
from docdebt.repair.repairer import RepairConfig, Repairer
from docdebt.repair.verifier import Verifier
from docdebt.settings import Settings


@dataclass
class DetectionPipeline:
    auditor: Auditor
    router: Router
    tracker: UsageTracker
    workers: int = 1

    def run_one(self, unit: MethodUnit) -> RoutedResult:
        report = self.auditor.audit(unit)
        return self.router.route(unit, report.findings)

    def run(self, units: list[MethodUnit]) -> list[RoutedResult]:
        """Detect debt in every unit; results keep the input order."""
        if self.workers <= 1:
            return [self.run_one(u) for u in units]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self.run_one, units))


def build_registry(settings: Settings) -> PromptRegistry:
    fewshot_path = settings.resolve(settings.fewshot_file)
    fewshot = FewShotSet.load(fewshot_path) if fewshot_path else None
    return PromptRegistry(fewshot=fewshot, overrides=settings.wording_overrides)


def build_detection_pipeline(
    settings: Settings,
    repo_root: str | Path | None = None,
    *,
    cascade: list[str] | None = None,
    factory: ChatModelFactory | None = None,
) -> DetectionPipeline:
    """Build the pipeline described by ``settings``.

    Args:
        settings: loaded configuration.
        repo_root: repository checkout, needed only when a judge uses context.
        cascade: override the configured cascade (e.g. a single model from the CLI).
        factory: LangChain model factory override (tests inject fakes here).
    """
    tracker = UsageTracker()
    registry = build_registry(settings)
    policy = settings.router.to_policy()
    if cascade:
        policy.cascade = list(cascade)

    specs = settings.model_specs()
    judges: dict[str, Judge] = {}
    for model_id in policy.cascade:
        client = LLMClient(get_model_spec(model_id, specs), factory=factory, tracker=tracker)
        judges[model_id] = Judge(client, registry, settings.judge_config(model_id))

    context_provider = None
    if any(j.config.use_context for j in judges.values()):
        if repo_root is None:
            raise ValueError("A judge uses context: pass the repository root.")
        context_provider = ContextRetriever(repo_root, settings.context.sources)

    return DetectionPipeline(
        auditor=Auditor(settings.auditor.to_config()),
        router=Router(judges, policy, tracker, context_provider),
        tracker=tracker,
        workers=settings.workers,
    )


def build_repairer(
    settings: Settings,
    model_id: str | None = None,
    *,
    pattern: str | None = None,
    wording: int | None = None,
    hint_findings: bool | None = None,
    factory: ChatModelFactory | None = None,
    tracker: UsageTracker | None = None,
) -> Repairer:
    """Build the Repairer described by ``settings``.

    Args:
        settings: loaded configuration.
        model_id: repair model; defaults to ``repair.model``, then to the first
            model of the router cascade.
        pattern: repair prompt pattern; defaults to ``repair.pattern``.
        wording: explicit wording index for this model and pattern (needed for
            models the research did not screen).
        hint_findings: overrides ``repair.hint_findings`` when not ``None``.
        factory: LangChain model factory override (tests inject fakes here).
        tracker: shared usage tracker.
    """
    cfg = settings.repair
    model_id = model_id or cfg.model or settings.router.cascade[0]
    pattern = pattern or cfg.pattern
    overrides = {m: dict(p) for m, p in cfg.wording_overrides.items()}
    if wording is not None:
        overrides.setdefault(model_id, {})[pattern] = wording
    fewshot_path = settings.resolve(cfg.fewshot_file)
    registry = RepairRegistry(
        fewshot=RepairFewShotSet.load(fewshot_path) if fewshot_path else None, overrides=overrides
    )
    client = LLMClient(get_model_spec(model_id, settings.model_specs()), factory=factory, tracker=tracker)
    return Repairer(
        client,
        registry,
        Verifier(settings.auditor.to_config()),
        RepairConfig(
            pattern=pattern,
            max_attempts=cfg.max_attempts,
            hint_findings=cfg.hint_findings if hint_findings is None else hint_findings,
        ),
    )
