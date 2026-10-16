"""HybridRepairer: the research hybrid Replacement mechanism behind the Repairer interface.

Four layers (see ``docdebt.repair.structural_layer``, ``semantic_triage``
and ``hybrid``):

1. deterministic structural plan: phantom / duplicate tags are deleted, a
   renamed parameter keeps its description, missing tags become WRITE tasks;
2. deterministic semantic triage: fixed rules flag suspicious fields;
3. one targeted model call: the model only writes missing fields and answers
   ``keep`` or a corrected text for the fields it is asked to judge;
4. deterministic post-validation: every answer is checked before it is
   applied, and the result is assembled as one span patch on the original,
   so every untouched line stays byte-identical.

It never asks the model what the code establishes with certainty, and never
lets it rewrite the whole block. A repair is proposed only if the patched
Javadoc passes the structural check and the Auditor.

Differences from the research run (all to make one-method-at-a-time service
calls more reliable): one item per call instead of batches of five; the
detector's verified citations are added as hints on the field they dispute;
optional retrieved project context; JSON-only answers requested from the
provider when it supports it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect

from docdebt.core.auditor import Auditor, AuditorConfig
from docdebt.core.javadoc_edit import DESCRIPTION, OTHER, PARAM, RETURN, THROWS, FieldChange, FieldRef
from docdebt.core.models import Finding, FindingKind, MethodUnit
from docdebt.llm.client import LLMClient, LLMError
from docdebt.repair import hybrid
from docdebt.repair.brace_tokens import encode_braces, parse_batched_json_patch
from docdebt.repair.plan import field_ref_from_text
from docdebt.repair.repairer import DEFAULT_REPAIR_TOKENS, Attempt, RepairResult, RepairStatus
from docdebt.repair.spans import parse_javadoc_spans
from docdebt.repair.structural_layer import _simple

MECHANISM = "hybrid"
_ITEM_ID = "m"


def prompt_version() -> str:
    """Version string of the hybrid prompt: changes whenever its source does."""
    source = inspect.getsource(hybrid.build_hybrid_prompt) + inspect.getsource(hybrid._item_block)
    return f"repair/hybrid/v1/{hashlib.sha256(source.encode()).hexdigest()[:8]}"


def _key_for(ref: FieldRef, parsed) -> str:
    """The spans key of the field a finding is about."""
    if ref.kind == DESCRIPTION:
        return "description"
    if ref.kind == RETURN:
        return "@return"
    if ref.kind == PARAM:
        return f"@param {ref.name}"
    if ref.kind == THROWS and parsed is not None:
        for f in parsed.fields:
            if f.kind == "throws" and _simple(f.name) == _simple(ref.name or ""):
                return f.key
    return f"@throws {ref.name}"


def _ref_for_key(key: str) -> FieldRef:
    if key == "description":
        return FieldRef(DESCRIPTION)
    if key.startswith("@return"):
        return FieldRef(RETURN)
    head, _, name = key.partition(" ")
    name = name.split("#")[0] or None
    if head == "@param":
        return FieldRef(PARAM, name)
    if head in ("@throws", "@exception"):
        return FieldRef(THROWS, name)
    return FieldRef(OTHER, head.lstrip("@"))


def _changes(original: str, repaired: str) -> list[FieldChange]:
    """Field-level differences between the original and the patched Javadoc."""
    before = {f.key: f.text for f in (parse_javadoc_spans(original).fields if original.strip() else [])}
    after = {f.key: f.text for f in parse_javadoc_spans(repaired).fields}
    out: list[FieldChange] = []
    for key, text in before.items():
        if key not in after:
            out.append(FieldChange(_ref_for_key(key), "remove", text, None))
        elif " ".join(after[key].split()) != " ".join(text.split()):
            out.append(FieldChange(_ref_for_key(key), "replace", text, after[key]))
    for key, text in after.items():
        if key not in before:
            out.append(FieldChange(_ref_for_key(key), "add", None, text))
    return out


def _cited_hints(unit: MethodUnit, findings: list[Finding], parsed) -> dict[str, list[str]]:
    """The detector's verified citations, as hints on the field they dispute."""
    hints: dict[str, list[str]] = {}
    for f in findings:
        if f.kind != FindingKind.SEMANTIC or not f.is_debt:
            continue
        key = _key_for(field_ref_from_text(f.field, unit), parsed)
        text = f.message
        if f.quoted_doc:
            text += f' The documentation says "{f.quoted_doc}"'
            if f.quoted_code:
                text += f', the code does "{f.quoted_code}"'
            text += "."
        hints.setdefault(key, []).append(text)
    return hints


class HybridRepairer:
    """Repair one method's Javadoc with the hybrid Replacement mechanism."""

    mechanism = MECHANISM

    def __init__(
        self,
        client: LLMClient,
        *,
        judge: str = "flagged",
        auditor_config: AuditorConfig | None = None,
        rename_rule: str = "name_sim",
    ) -> None:
        if judge not in hybrid.JUDGE_VARIANTS:
            raise ValueError(f"judge must be one of {hybrid.JUDGE_VARIANTS}")
        self.client = client
        self.judge = judge
        self.rename_rule = rename_rule
        self.auditor = Auditor(auditor_config or AuditorConfig())

    @property
    def model_id(self) -> str:
        return self.client.spec.id

    # -- one model call --------------------------------------------------------

    def _ask(self, it: dict, context: str | None, attempt: Attempt) -> dict:
        budget = self.client.spec.max_tokens.get("repair", DEFAULT_REPAIR_TOKENS)
        prompt = hybrid.build_hybrid_prompt([it], context=context)
        try:
            response = self.client.invoke(prompt, purpose="repair", max_tokens=budget, json_mode=True)
        except LLMError as exc:
            attempt.error = str(exc)
            return {}
        attempt.raw, attempt.truncated = response.text, response.truncated
        patch = parse_batched_json_patch(response.text)
        if not isinstance(patch, dict):
            return {}
        item = patch.get(_ITEM_ID)
        if isinstance(item, dict):
            return item
        # A model answering a single item sometimes drops the item-id level.
        if len(patch) == 1 and isinstance(next(iter(patch.values())), dict):
            return next(iter(patch.values()))
        return patch

    # -- the Repairer interface ---------------------------------------------------

    def repair(self, unit: MethodUnit, findings: list[Finding], context: str | None = None) -> RepairResult:
        original = unit.javadoc or ""
        result = RepairResult(unit.id, RepairStatus.NOT_NEEDED, original, model_id=self.model_id)
        result.mechanism = MECHANISM
        result.prompt_version = prompt_version()
        if not any(f.is_debt for f in findings):
            result.reason = "no_debt_finding"
            return result
        if not original.strip():
            result.status, result.reason = RepairStatus.SKIPPED, "no_existing_javadoc"
            return result

        it = hybrid.prepare_hybrid_item(
            _ITEM_ID, unit.code, original, judge=self.judge, rename_rule=self.rename_rule
        )
        if it is None:
            result.status, result.reason = RepairStatus.FAILED, "method_code_not_parsable"
            return result

        # The detector's citations: the disputed field is always judged, with the citation as a hint.
        for key, notes in _cited_hints(unit, findings, it["parsed"]).items():
            task = next((j for j in it["judge"] if j["key"] == key), None)
            if task is None:
                task = next((c for c in it["context"] if c["key"] == key), None)
                if task is not None:
                    it["context"].remove(task)
                    it["judge"].append(task)
            if task is not None:
                task["hints"] = [*task["hints"], *(encode_braces(n) for n in notes)]

        written: dict[str, str] = {}
        replaced: dict[str, str] = {}
        if it["write"] or it["judge"]:
            attempt = Attempt(1)
            result.attempts.append(attempt)
            answer = self._ask(it, context, attempt)
            written, replaced, log, failed = hybrid.validate_item_patch(it, answer)
            result.field_log = log
            if failed:
                # One cheap retry restricted to the fields the model failed to write.
                retry = Attempt(2)
                result.attempts.append(retry)
                reduced = hybrid.retry_item_for_writes(it, failed)
                w2, _, log2, _ = hybrid.validate_item_patch(reduced, self._ask(reduced, context, retry))
                written.update(w2)
                result.field_log += [
                    dict(e, outcome=f"{e['outcome']}_after_retry") if e["outcome"] != "write_failed" else e
                    for e in log2
                ]
            if result.llm_failed:
                result.status, result.reason = RepairStatus.FAILED, "llm_error"
                return result

        repaired = hybrid.assemble_hybrid_doc(it, written, replaced)
        ok, problems = hybrid.check_structure(repaired, it["facts"])
        audit = self.auditor.audit(dataclasses.replace(unit, javadoc=repaired))
        if not ok or audit.has_structural_debt:
            open_problems = sorted(k for k, v in problems.items() if v) or sorted({f.rule for f in audit.errors})
            result.status, result.reason = RepairStatus.FAILED, "structure:" + ",".join(open_problems)
            result.unresolved = open_problems
            return result

        changes = _changes(original, repaired)
        if not changes:
            result.status, result.reason = RepairStatus.UNCHANGED, "model_kept_disputed_fields"
            return result
        result.status, result.repaired, result.changes = RepairStatus.VERIFIED, repaired, changes
        return result
