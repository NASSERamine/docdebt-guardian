"""Graph state schema and the statuses that flow through it.

Every field is a JSON-safe primitive (see ``docdebt.core.serialize``), never
a dataclass instance: LangGraph's checkpointer serializes state to disk at
every step, and only plain dicts/lists/strings/numbers survive that reliably
across process restarts and library versions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NotRequired, TypedDict


class DetectionOutcome(StrEnum):
    DEBT = "debt"
    CLEAN = "clean"
    ABSTAIN = "abstain"
    SKIPPED = "skipped"  # no judge configured: structural check only


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class FinalOutcome(StrEnum):
    NO_DEBT = "no_debt"  # nothing to fix
    REPAIR_NOT_PROPOSED = "repair_not_proposed"  # debt found, no safe fix to propose
    ACCEPTED = "accepted"  # a verified repair, accepted by the reviewer
    REJECTED = "rejected"  # a verified repair, rejected by the reviewer


class DocDebtState(TypedDict):
    """State carried through the graph for one method.

    ``unit`` is the JSON-safe form of the :class:`~docdebt.core.models.MethodUnit`
    being analysed (see ``docdebt.core.serialize.unit_to_dict``). Every other
    field is populated as the method moves through the graph; fields the
    method never reaches keep their initial value.
    """

    unit: dict
    content_fingerprint: NotRequired[str]

    structural_findings: list[dict]
    has_structural_debt: bool

    detection_status: NotRequired[str]  # DetectionOutcome
    detection_reason: NotRequired[str]
    detection_model: NotRequired[str | None]
    detection_confidence: NotRequired[float | None]
    semantic_findings: NotRequired[list[dict]]
    detection_error: NotRequired[str | None]  # the model could not be called (key, quota, network)

    repair_status: NotRequired[str]  # docdebt.repair.repairer.RepairStatus
    repair_reason: NotRequired[str]
    repair_model: NotRequired[str]
    repair_prompt_version: NotRequired[str]
    repaired_doc: NotRequired[str | None]
    changes: NotRequired[list[dict]]
    unresolved: NotRequired[list[str]]
    repair_attempts: NotRequired[int]
    repair_mechanism: NotRequired[str]  # "hybrid" | "regeneration"
    field_log: NotRequired[list[dict]]  # hybrid mechanism: per-field post-validation outcome
    rag_context: NotRequired[
        list[dict]
    ]  # retrieved excerpts given to the model (source, label, path, score, text)

    review_decision: NotRequired[str | None]  # ReviewDecision, set by the human
    final: NotRequired[str]  # FinalOutcome
