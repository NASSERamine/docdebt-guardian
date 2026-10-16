"""Learning from reviewers' decisions.

Every accepted or rejected proposal is a free label: accepted means the tool
was right, rejected means it was wrong or its fix was not good enough. Two
uses, both read-only (nothing is changed automatically):

* ``precision``: accepted / (accepted + rejected), overall and broken down
  by debt kind, detection model, repair mechanism and reject reason;
* ``threshold_suggestion``: for semantic debt, the precision reached at
  each confidence threshold of the Judge, and the lowest threshold that
  reaches a target precision. The dashboard shows it; a person applies it.

Expired proposals are excluded: nobody decided on them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from docdebt.service.store import ReviewRequest

MIN_DECISIONS_PRECISION = 10
MIN_DECISIONS_THRESHOLD = 20
DEFAULT_TARGET = 0.8


def _rate(accepted: int, rejected: int) -> float | None:
    total = accepted + rejected
    return round(accepted / total, 3) if total else None


def _breakdown(decisions: list[ReviewRequest], key) -> list[dict]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for d in decisions:
        groups[key(d) or "unknown"][d.decision] += 1
    return sorted(
        (
            {
                "key": name,
                "accepted": c["accept"],
                "rejected": c["reject"],
                "precision": _rate(c["accept"], c["reject"]),
            }
            for name, c in groups.items()
        ),
        key=lambda row: -(row["accepted"] + row["rejected"]),
    )


def precision(decisions: Iterable[ReviewRequest]) -> dict:
    decisions = [d for d in decisions if d.decision in ("accept", "reject")]
    accepted = sum(d.decision == "accept" for d in decisions)
    rejected = len(decisions) - accepted
    return {
        "decisions": len(decisions),
        "accepted": accepted,
        "rejected": rejected,
        "precision": _rate(accepted, rejected),
        "enough": len(decisions) >= MIN_DECISIONS_PRECISION,
        "min_decisions": MIN_DECISIONS_PRECISION,
        "by_kind": _breakdown(decisions, lambda d: d.debt_kind),
        "by_model": _breakdown(decisions, lambda d: d.detection_model or d.repair_model),
        "by_mechanism": _breakdown(decisions, lambda d: d.repair_mechanism),
        "reject_reasons": dict(
            Counter(d.reject_reason or "unspecified" for d in decisions if d.decision == "reject")
        ),
    }


def threshold_suggestion(
    decisions: Iterable[ReviewRequest], target: float = DEFAULT_TARGET, current: float | None = None
) -> dict:
    """Precision of semantic-debt proposals at each confidence threshold.

    A proposal with confidence ``c`` would still have been made with a
    threshold ``t <= c``. For each candidate threshold (the distinct
    confidences observed), the curve gives how many proposals would remain
    and their precision. The suggestion is the lowest threshold whose
    precision reaches ``target`` (lowest, so as few true debts as possible
    are lost), among thresholds that keep at least 5 decisions.
    """
    semantic = [
        d
        for d in decisions
        if d.decision in ("accept", "reject") and d.confidence is not None and d.debt_kind in ("semantic", "both")
    ]
    curve = []
    for t in sorted({round(d.confidence, 2) for d in semantic}):
        kept = [d for d in semantic if d.confidence >= t - 1e-9]
        accepted = sum(d.decision == "accept" for d in kept)
        curve.append(
            {
                "threshold": t,
                "kept": len(kept),
                "accepted": accepted,
                "precision": _rate(accepted, len(kept) - accepted),
            }
        )
    suggested = next(
        (row["threshold"] for row in curve if row["kept"] >= 5 and (row["precision"] or 0) >= target), None
    )
    enough = len(semantic) >= MIN_DECISIONS_THRESHOLD
    return {
        "decisions": len(semantic),
        "enough": enough,
        "min_decisions": MIN_DECISIONS_THRESHOLD,
        "target": target,
        "current": current,
        "curve": curve,
        "suggested": suggested if enough else None,
    }
