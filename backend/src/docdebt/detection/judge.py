"""Judge: semantic verdict on one method, with confidence and a validated citation.

How a verdict is reached:

1. **Votes.** The method is judged with one or more validated detection
   prompts (the model's locked wording for each configured pattern),
   optionally sampled several times. Each answer is parsed into debt,
   clean or no verdict.
2. **Confidence.** Confidence is the share of valid votes that agree with
   the majority. A tie or a lack of valid votes gives an abstention. With
   a single vote at temperature 0 the confidence is 1.0 by construction,
   so use several patterns or samples when you need graded confidence.
3. **Citation.** When the majority says debt, the citation prompt asks the
   model to quote the contradicted claim and the contradicting code. Only
   quotes found verbatim become semantic findings.
4. **Decision.** Debt needs support: at least one validated citation, or
   structural errors already found by the Auditor. A debt vote with
   neither is reported as an abstention (``unsupported_verdict``).
   Finally, a confidence below the threshold becomes an abstention too.

The Judge never raises on bad model output: unparsable answers and failed
calls become abstentions, recorded in ``votes`` for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from docdebt.core.models import Finding, FindingKind, MethodUnit, Severity
from docdebt.detection.citations import validate_citations
from docdebt.llm.client import LLMClient, LLMError
from docdebt.llm.parsing import parse_binary_verdict, parse_claim_citations
from docdebt.llm.prompts.explain import EXPLAIN_VERSION, build_explain_prompt
from docdebt.llm.prompts.registry import PromptRegistry


class Status(StrEnum):
    DEBT = "debt"
    CLEAN = "clean"
    ABSTAIN = "abstain"


@dataclass
class JudgeConfig:
    """Judge settings for one model.

    Attributes:
        patterns: validated prompt patterns to vote with, e.g.
            ``["zeroshot"]`` or ``["zeroshot", "cot"]``.
        samples_per_pattern: answers drawn per pattern. With more than one,
            ``sample_temperature`` is used instead of the model temperature.
        sample_temperature: temperature for multi-sample voting.
        explain: run the citation prompt on debt votes.
        require_citation: a debt vote without validated citation and
            without structural errors becomes an abstention.
        confidence_threshold: below this confidence the result is an
            abstention. Calibrate it with ``scripts/calibrate.py``.
        use_context: pass retrieved context to the detection prompts.
    """

    patterns: list[str] = field(default_factory=lambda: ["zeroshot"])
    samples_per_pattern: int = 1
    sample_temperature: float = 0.7
    explain: bool = True
    require_citation: bool = True
    confidence_threshold: float = 0.0
    use_context: bool = False


@dataclass
class Vote:
    pattern: str
    sample: int
    verdict: bool | None
    raw: str
    prompt_version: str
    truncated: bool = False
    error: str | None = None


@dataclass
class JudgeResult:
    """Outcome of judging one method with one model."""

    method_id: str
    model_id: str
    status: Status
    confidence: float
    reason: str
    votes: list[Vote] = field(default_factory=list)
    semantic_findings: list[Finding] = field(default_factory=list)
    explanation_raw: str | None = None
    rejected_citations: int = 0

    @property
    def debt_votes(self) -> int:
        return sum(1 for v in self.votes if v.verdict is True)

    @property
    def clean_votes(self) -> int:
        return sum(1 for v in self.votes if v.verdict is False)

    @property
    def valid_votes(self) -> int:
        return self.debt_votes + self.clean_votes


class Judge:
    """Semantic judge bound to one model."""

    def __init__(
        self,
        client: LLMClient,
        registry: PromptRegistry,
        config: JudgeConfig | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config or JudgeConfig()

    @property
    def model_id(self) -> str:
        return self.client.spec.id

    # -- voting ------------------------------------------------------------

    def _votes(self, unit: MethodUnit, context: str | None) -> list[Vote]:
        cfg = self.config
        votes: list[Vote] = []
        temperature = cfg.sample_temperature if cfg.samples_per_pattern > 1 else None
        for pattern in cfg.patterns:
            prompt = self.registry.build(
                self.model_id,
                pattern,
                unit.code,
                unit.javadoc or "",
                context=context if cfg.use_context else None,
            )
            for sample in range(cfg.samples_per_pattern):
                try:
                    response = self.client.invoke(prompt.text, purpose=pattern, temperature=temperature)
                except LLMError as exc:
                    votes.append(Vote(pattern, sample, None, "", prompt.version, error=str(exc)))
                    continue
                votes.append(
                    Vote(
                        pattern=pattern,
                        sample=sample,
                        verdict=parse_binary_verdict(response.text),
                        raw=response.text,
                        prompt_version=prompt.version,
                        truncated=response.truncated,
                    )
                )
        return votes

    # -- citation ----------------------------------------------------------

    def _explain(self, unit: MethodUnit, confidence: float) -> tuple[list[Finding], str | None, int]:
        prompt = build_explain_prompt(unit.code, unit.javadoc or "")
        try:
            response = self.client.invoke(prompt, purpose="explain")
        except LLMError:
            return [], None, 0
        citations = parse_claim_citations(response.text) or []
        checked = validate_citations(citations, unit.javadoc or "", unit.code)
        findings = [
            Finding(
                kind=FindingKind.SEMANTIC,
                severity=Severity.ERROR,
                rule="contradicted-claim",
                message=c.citation.reason or "The documentation contradicts the code.",
                method_id=unit.id,
                field=c.citation.field or "description",
                line=unit.javadoc_start_line,
                quoted_doc=c.citation.claim,
                quoted_code=c.citation.code,
                confidence=confidence,
                model=self.model_id,
                prompt_version=EXPLAIN_VERSION,
            )
            for c in checked
            if c.valid
        ]
        return findings, response.text, sum(1 for c in checked if not c.valid)

    # -- decision ----------------------------------------------------------

    def judge(
        self,
        unit: MethodUnit,
        structural_findings: list[Finding] | None = None,
        context: str | None = None,
    ) -> JudgeResult:
        """Judge one method.

        Args:
            unit: the method, with its Javadoc.
            structural_findings: Auditor findings for the same method; their
                errors count as support for a debt vote.
            context: retrieved project context (used only if the config
                enables it).
        """
        votes = self._votes(unit, context)
        result = JudgeResult(unit.id, self.model_id, Status.ABSTAIN, 0.0, "", votes=votes)

        if result.valid_votes == 0:
            result.reason = "no_valid_vote"
            return result
        if result.debt_votes == result.clean_votes:
            result.reason = "tie"
            result.confidence = 0.5
            return result

        majority_debt = result.debt_votes > result.clean_votes
        confidence = max(result.debt_votes, result.clean_votes) / result.valid_votes
        result.confidence = confidence

        if not majority_debt:
            result.status, result.reason = Status.CLEAN, "majority_clean"
        else:
            has_structural = any(f.is_debt for f in structural_findings or [])
            if self.config.explain:
                findings, raw, rejected = self._explain(unit, confidence)
                result.semantic_findings = findings
                result.explanation_raw = raw
                result.rejected_citations = rejected
            if result.semantic_findings:
                result.status, result.reason = Status.DEBT, "cited_contradiction"
            elif has_structural:
                result.status, result.reason = Status.DEBT, "structural_support"
            elif self.config.require_citation and self.config.explain:
                result.status, result.reason = Status.ABSTAIN, "unsupported_verdict"
            else:
                result.status, result.reason = Status.DEBT, "majority_debt"

        if result.status != Status.ABSTAIN and confidence < self.config.confidence_threshold:
            result.status, result.reason = Status.ABSTAIN, "low_confidence"
        return result
