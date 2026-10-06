"""Repairer: propose a verified fix for the Javadoc of one method.

For one method and its debt findings:

1. an edit plan lists the fields that may change;
2. the model receives the research Regeneration prompt (a complete corrected
   Javadoc is requested);
3. only the planned fields are read from its answer and merged into the
   original, which keeps everything else byte for byte;
4. the Verifier checks the merged text; if it fails, the model gets the
   reasons and tries again, up to ``max_attempts`` times.

A repair the Verifier rejects is never returned as a proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from docdebt.core.javadoc_edit import EditPlan, FieldChange, MergeResult, apply_edits
from docdebt.core.models import Finding, MethodUnit
from docdebt.llm.client import LLMClient, LLMError
from docdebt.repair.extract import extract_javadoc_block
from docdebt.repair.plan import plan_from_findings
from docdebt.repair.registry import RepairRegistry
from docdebt.repair.verifier import Issue, Verifier

DEFAULT_REPAIR_TOKENS = 1400


class RepairStatus(StrEnum):
    VERIFIED = "verified"  # a checked fix is ready to propose
    UNCHANGED = "unchanged"  # the model kept every disputed field: nothing to propose
    FAILED = "failed"  # no attempt passed verification
    NOT_NEEDED = "not_needed"  # no debt finding, nothing to repair
    SKIPPED = "skipped"  # the method has no Javadoc to repair


@dataclass
class RepairConfig:
    """Repairer settings.

    Attributes:
        pattern: repair prompt pattern (zeroshot, oneshot, threeshot, cot).
        max_attempts: model calls per method (each retry carries the reasons
            for the previous rejection).
        hint_findings: list the known problems in the first prompt. Off by
            default so the first attempt is exactly the research prompt.
    """

    pattern: str = "zeroshot"
    max_attempts: int = 2
    hint_findings: bool = False


@dataclass
class Attempt:
    number: int
    raw: str = ""
    block: str | None = None
    merged: str | None = None
    issues: list[Issue] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None


@dataclass
class RepairResult:
    method_id: str
    status: RepairStatus
    original: str
    repaired: str | None = None
    changes: list[FieldChange] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    plan: EditPlan = field(default_factory=EditPlan)
    model_id: str = ""
    prompt_version: str = ""
    reason: str = ""

    @property
    def llm_failed(self) -> bool:
        """True when no model call succeeded (server down, no credit...)."""
        return bool(self.attempts) and all(a.error for a in self.attempts)


def _hints(findings: list[Finding]) -> list[str]:
    out = []
    for f in findings:
        if not f.is_debt:
            continue
        text = f.message
        if f.quoted_doc:
            text += f' Statement: "{f.quoted_doc}". Code: "{f.quoted_code}".'
        out.append(text)
    return out


class Repairer:
    def __init__(
        self,
        client: LLMClient,
        registry: RepairRegistry,
        verifier: Verifier | None = None,
        config: RepairConfig | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.verifier = verifier or Verifier()
        self.config = config or RepairConfig()

    @property
    def model_id(self) -> str:
        return self.client.spec.id

    def repair(self, unit: MethodUnit, findings: list[Finding]) -> RepairResult:
        original = unit.javadoc or ""
        plan = plan_from_findings(unit, findings)
        result = RepairResult(unit.id, RepairStatus.NOT_NEEDED, original, plan=plan, model_id=self.model_id)
        if plan.is_empty:
            result.reason = "no_debt_finding"
            return result
        if not original.strip():
            result.status, result.reason = RepairStatus.SKIPPED, "no_existing_javadoc"
            return result

        cfg = self.config
        budget = self.client.spec.max_tokens.get("repair", DEFAULT_REPAIR_TOKENS)
        feedback: list[str] = []
        last_issues: list[Issue] = []
        for number in range(1, cfg.max_attempts + 1):
            prompt = self.registry.build(
                self.model_id,
                cfg.pattern,
                unit.code,
                original,
                hints=_hints(findings) if cfg.hint_findings and number == 1 else None,
                feedback=feedback,
            )
            result.prompt_version = prompt.version
            attempt = Attempt(number)
            result.attempts.append(attempt)
            try:
                response = self.client.invoke(prompt.text, purpose="repair", max_tokens=budget)
            except LLMError as exc:
                attempt.error = str(exc)
                continue
            attempt.raw, attempt.truncated = response.text, response.truncated
            if response.truncated:
                attempt.issues = last_issues = [Issue("truncated", None, "The answer was cut off.")]
                feedback = ["The answer was cut off: be concise and finish the Javadoc block."]
                continue
            block = extract_javadoc_block(response.text)
            attempt.block = block
            if block is None:
                attempt.issues = last_issues = [Issue("no_javadoc", None, "No Javadoc block found in the answer.")]
                feedback = ["Answer with a complete Javadoc block that starts with /** and ends with */."]
                continue

            merge: MergeResult = apply_edits(original, block, plan, unit.signature, base_indent=unit.indent)
            attempt.merged = merge.text
            issues = self.verifier.verify(unit, original, plan, findings, merge)
            attempt.issues = issues
            last_issues = issues
            if issues:
                feedback = [i.message for i in issues]
                continue

            result.unresolved = [r.label for r in merge.unchanged]
            if merge.changes:
                result.status, result.repaired, result.changes = (
                    RepairStatus.VERIFIED,
                    merge.text,
                    merge.changes,
                )
            else:
                result.status, result.reason = RepairStatus.UNCHANGED, "model_kept_disputed_fields"
            return result

        result.status = RepairStatus.FAILED
        result.reason = (
            "llm_error"
            if result.llm_failed
            else ",".join(sorted({i.code for i in last_issues})) or "no_usable_answer"
        )
        return result
