"""Graph nodes: each one calls an already-validated component and returns a
partial state update. No business logic lives here — see ``docdebt.core.auditor``,
``docdebt.detection.router`` and ``docdebt.repair.repairer`` for that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt

from docdebt.core.auditor import Auditor
from docdebt.core.models import Finding
from docdebt.core.serialize import finding_from_dict, unit_from_dict
from docdebt.detection.router import Router
from docdebt.graph.state import DetectionOutcome, DocDebtState, FinalOutcome, ReviewDecision
from docdebt.repair.repairer import Repairer, RepairStatus


@dataclass
class GraphDeps:
    """The components a compiled graph calls. Built once per graph, reused for
    every method it processes. ``router`` is ``None`` for a structural-only
    graph (the Judge is never called).

    ``repairers`` maps a model id to the repairer that uses it: a debt found
    by a model of the cascade is repaired by that same model; ``repairer`` is
    the default (the first model of the cascade). ``rag`` builds the
    retrieved project context (``docdebt.rag.retriever.RagProvider``);
    ``rag_repair`` / ``rag_detection`` say where it is given to the model.
    """

    auditor: Auditor
    router: Router | None
    repairer: Repairer | None
    repairers: dict[str, Any] = field(default_factory=dict)
    rag: Any = None
    rag_repair: bool = False
    rag_detection: bool = False


def content_fingerprint(unit) -> str:
    """Hash of the exact content the graph reasons about (code and Javadoc).

    Used by :meth:`~docdebt.graph.build.GraphApp.run_unit` to tell an unchanged
    method (safe to reuse a prior finished run for, without spending another
    LLM call) from one whose signature is unchanged but whose code or Javadoc
    was edited since the last run (which must be re-analysed).
    """
    return hashlib.sha256(f"{unit.code}\0{unit.javadoc or ''}".encode()).hexdigest()


def _findings(state: DocDebtState) -> list[Finding]:
    structural = [finding_from_dict(f) for f in state["structural_findings"]]
    semantic = [finding_from_dict(f) for f in state.get("semantic_findings", [])]
    return structural + semantic


def make_audit_node(deps: GraphDeps):
    def audit(state: DocDebtState) -> dict:
        unit = unit_from_dict(state["unit"])
        report = deps.auditor.audit(unit)
        return {
            "structural_findings": [f.to_dict() for f in report.findings],
            "has_structural_debt": report.has_structural_debt,
            "content_fingerprint": content_fingerprint(unit),
        }

    return audit


def make_detect_node(deps: GraphDeps):
    def detect(state: DocDebtState) -> dict:
        if deps.router is None:
            return {"detection_status": DetectionOutcome.SKIPPED.value, "semantic_findings": []}
        unit = unit_from_dict(state["unit"])
        structural = [finding_from_dict(f) for f in state["structural_findings"]]
        routed = deps.router.route(unit, structural)
        rag_context = (
            [i.to_dict() for i in deps.rag.build(unit).items]
            if deps.rag is not None and deps.rag_detection
            else []
        )
        votes = routed.final.votes if routed.final else []
        llm_error = next((v.error for v in votes if v.error), None)
        return {
            "rag_context": rag_context,
            "detection_error": llm_error if routed.final and routed.final.valid_votes == 0 else None,
            "detection_status": routed.status.value,
            "detection_reason": routed.final.reason if routed.final else routed.note,
            "detection_model": routed.final.model_id if routed.final else None,
            "detection_confidence": routed.final.confidence if routed.final else None,
            "semantic_findings": [f.to_dict() for f in (routed.final.semantic_findings if routed.final else [])],
        }

    return detect


def has_debt(state: DocDebtState) -> bool:
    """Debt when an exact rule fails OR the model says so (one detection on the whole Javadoc)."""
    if state["has_structural_debt"]:
        return True
    if state.get("detection_status") == DetectionOutcome.DEBT.value:
        return True
    return bool(state.get("semantic_findings"))


def make_repair_node(deps: GraphDeps):
    def repair(state: DocDebtState) -> dict:
        unit = unit_from_dict(state["unit"])
        repairer = deps.repairers.get(state.get("detection_model") or "") or deps.repairer
        if repairer is None:  # no model configured: the exact rules ran, nothing can be proposed
            return {"repair_status": RepairStatus.FAILED.value, "repair_reason": "no_model_configured"}
        context = None
        rag_context = list(state.get("rag_context", []))
        if deps.rag is not None and deps.rag_repair:
            retrieved = deps.rag.build(unit)
            context = retrieved.text or None
            rag_context = [i.to_dict() for i in retrieved.items]
        result = repairer.repair(unit, _findings(state), context=context)
        return {
            "rag_context": rag_context,
            "repair_mechanism": getattr(result, "mechanism", "regeneration"),
            "field_log": list(getattr(result, "field_log", [])),
            "repair_status": result.status.value,
            "repair_reason": result.reason,
            "repair_model": result.model_id,
            "repair_prompt_version": result.prompt_version,
            "repaired_doc": result.repaired,
            "changes": [
                {"field": c.ref.label, "action": c.action, "before": c.before, "after": c.after}
                for c in result.changes
            ],
            "unresolved": result.unresolved,
            "repair_attempts": len(result.attempts),
        }

    return repair


def is_verified(state: DocDebtState) -> bool:
    return state.get("repair_status") == RepairStatus.VERIFIED.value


def review(state: DocDebtState) -> dict:
    """Pause and wait for a human decision on a verified repair.

    The value passed to :func:`interrupt` is what a caller sees while paused
    (``get_state(...).tasks[0].interrupts``); resuming with
    ``Command(resume="accept")`` or ``Command(resume="reject")`` continues
    this same node with that value as the return of ``interrupt(...)``.
    """
    decision = interrupt(
        {
            "unit": state["unit"],
            "repaired_doc": state["repaired_doc"],
            "changes": state["changes"],
            "unresolved": state.get("unresolved", []),
        }
    )
    accepted = str(decision).strip().lower() == ReviewDecision.ACCEPT.value
    return {
        "review_decision": ReviewDecision.ACCEPT.value if accepted else ReviewDecision.REJECT.value,
        "final": FinalOutcome.ACCEPTED.value if accepted else FinalOutcome.REJECTED.value,
    }


def finalize_no_debt(_state: DocDebtState) -> dict:
    return {"final": FinalOutcome.NO_DEBT.value}


def finalize_not_proposed(_state: DocDebtState) -> dict:
    return {"final": FinalOutcome.REPAIR_NOT_PROPOSED.value}
