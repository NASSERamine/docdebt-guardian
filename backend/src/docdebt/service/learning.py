"""Learning from reviewers' decisions, as plain statistics.

Every accepted or rejected proposal is a free label: accepted means the tool
was right, rejected means it was wrong or its fix was not good enough. The
application shows the observed precision, accepted / (accepted + rejected),
overall and broken down by model, by the mechanism finally committed and by
reject reason. Nothing is ever changed from these numbers, and no threshold
applies to them.

Expired proposals are excluded: nobody decided on them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from docdebt.service.store import ReviewRequest


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
        key=lambda row: (-(row["accepted"] + row["rejected"]), row["key"]),
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
        "by_model": _breakdown(decisions, lambda d: d.detection_model or d.repair_model),
        # What was finally committed (replacement, regeneration or the reviewer's edit) for an
        # accept; the mechanism that produced the proposal for a reject.
        "by_mechanism": _breakdown(decisions, lambda d: d.accepted_mechanism or d.repair_mechanism),
        "reject_reasons": dict(
            Counter(d.reject_reason or "unspecified" for d in decisions if d.decision == "reject")
        ),
    }
