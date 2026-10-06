"""Router: cascade of judges, from the cheapest model to the strongest.

Research results behind it: local 7B models are free and private but
collapse toward a constant answer on some prompts, while cloud models
detect far better. The Router therefore asks the models of the cascade in
order and stops at the first confident, supported answer. It escalates to
the next model only when the current one abstains, the next model is
allowed by the privacy policy, and the budget is not exhausted.

The Auditor always runs first (outside the Router); its structural
errors are passed to every judge as support for a debt verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from docdebt.core.models import Finding, MethodUnit
from docdebt.detection.judge import Judge, JudgeResult, Status
from docdebt.llm.client import UsageTracker

ContextProvider = Callable[[MethodUnit], str]


@dataclass
class RouterPolicy:
    """Cascade and limits.

    Attributes:
        cascade: model ids, cheapest first.
        allow_cloud: when ``False``, non-local models are never called.
        max_tokens_total: stop escalating once this many tokens were spent
            across the run (``None`` = no limit).
        max_usd_total: same, in USD (needs prices in the model specs).
        skip_llm_when_structural: return straight after the Auditor when it
            already found structural errors, saving LLM calls. Semantic
            findings for those methods are then not computed.
    """

    cascade: list[str]
    allow_cloud: bool = False
    max_tokens_total: int | None = None
    max_usd_total: float | None = None
    skip_llm_when_structural: bool = False


@dataclass
class RoutedResult:
    """Final detection outcome for one method."""

    unit: MethodUnit
    structural_findings: list[Finding]
    final: JudgeResult | None
    trail: list[JudgeResult] = field(default_factory=list)
    skipped_models: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def has_structural_debt(self) -> bool:
        return any(f.is_debt for f in self.structural_findings)

    @property
    def semantic_status(self) -> Status:
        """The LLM's own answer, ``ABSTAIN`` when no model was asked."""
        return self.final.status if self.final is not None else Status.ABSTAIN

    @property
    def status(self) -> Status:
        """Final status. Structural errors are exact, so they always mean debt:
        an LLM answering "clean" can never cancel them."""
        if self.has_structural_debt:
            return Status.DEBT
        return self.semantic_status

    @property
    def findings(self) -> list[Finding]:
        semantic = self.final.semantic_findings if self.final else []
        return [*self.structural_findings, *semantic]

    @property
    def escalated(self) -> bool:
        return len(self.trail) > 1


class Router:
    """Run the judges of the cascade for one method at a time."""

    def __init__(
        self,
        judges: dict[str, Judge],
        policy: RouterPolicy,
        tracker: UsageTracker | None = None,
        context_provider: ContextProvider | None = None,
    ) -> None:
        missing = [m for m in policy.cascade if m not in judges]
        if missing:
            raise ValueError(f"No judge configured for cascade models: {missing}")
        self.judges = judges
        self.policy = policy
        self.tracker = tracker or UsageTracker()
        self.context_provider = context_provider

    def _budget_left(self) -> bool:
        p = self.policy
        if p.max_tokens_total is not None and self.tracker.total_tokens >= p.max_tokens_total:
            return False
        return not (p.max_usd_total is not None and self.tracker.cost_usd >= p.max_usd_total)

    def _allowed(self, model_id: str) -> bool:
        return self.policy.allow_cloud or self.judges[model_id].client.spec.is_local

    def route(self, unit: MethodUnit, structural_findings: list[Finding]) -> RoutedResult:
        result = RoutedResult(unit=unit, structural_findings=structural_findings, final=None)
        if self.policy.skip_llm_when_structural and any(f.is_debt for f in structural_findings):
            result.note = "structural_only"
            return result

        context_cache: str | None = None
        for model_id in self.policy.cascade:
            if not self._allowed(model_id):
                result.skipped_models.append(model_id)
                continue
            if result.trail and not self._budget_left():
                result.note = "budget_exhausted"
                break
            judge = self.judges[model_id]
            context = None
            if judge.config.use_context and self.context_provider is not None:
                if context_cache is None:
                    context_cache = self.context_provider(unit)
                context = context_cache
            outcome = judge.judge(unit, structural_findings, context=context)
            result.trail.append(outcome)
            result.final = outcome
            if outcome.status != Status.ABSTAIN:
                break
        if result.final is None and not result.note:
            result.note = "no_allowed_model"
        return result
