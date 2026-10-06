"""Threshold calibration and paired comparison statistics.

Calibration picks the confidence threshold under which the Judge abstains:
the smallest threshold whose precision on debt alerts reaches the target on
the *development* split of the benchmark. Always check the chosen threshold
on the frozen test split afterwards; choosing and reporting on the same
items overstates quality (selection bias seen in the research screening).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class CalibrationRecord:
    """One benchmark item as judged: true label, predicted status, confidence."""

    item_id: str
    is_debt: bool
    status: str  # "debt" | "clean" | "abstain"
    confidence: float


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    coverage: float  # share of items with a debt/clean answer
    precision: float | None  # correct debt alerts / debt alerts
    recall: float | None  # debt alerts on debt items / debt items (abstentions count as missed)
    specificity: float | None  # clean answers on clean items / clean items
    accuracy_answered: float | None
    n_alerts: int


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def metrics_at_threshold(records: list[CalibrationRecord], threshold: float) -> ThresholdMetrics:
    """Metrics when every answer with confidence below ``threshold`` becomes an abstention."""
    tp = fp = tn = fn_ans = answered = 0
    n_debt = sum(r.is_debt for r in records)
    n_clean = len(records) - n_debt
    for r in records:
        status = r.status if r.confidence >= threshold else "abstain"
        if status == "abstain":
            continue
        answered += 1
        if status == "debt":
            tp += r.is_debt
            fp += not r.is_debt
        else:
            tn += not r.is_debt
            fn_ans += r.is_debt
    return ThresholdMetrics(
        threshold=threshold,
        coverage=answered / len(records) if records else 0.0,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, n_debt),
        specificity=_ratio(tn, n_clean),
        accuracy_answered=_ratio(tp + tn, answered),
        n_alerts=tp + fp,
    )


def threshold_table(records: list[CalibrationRecord]) -> list[ThresholdMetrics]:
    """Metrics at every distinct confidence value observed (plus 0.0)."""
    thresholds = sorted({0.0, *(r.confidence for r in records)})
    return [metrics_at_threshold(records, t) for t in thresholds]


def choose_threshold(
    records: list[CalibrationRecord],
    target_precision: float,
    min_coverage: float = 0.0,
    min_alerts: int = 10,
) -> ThresholdMetrics | None:
    """Smallest threshold reaching ``target_precision`` with enough coverage and alerts.

    Returns ``None`` when no threshold qualifies; the model should then not
    be used alone for semantic alerts.
    """
    for row in threshold_table(records):
        if (
            row.precision is not None
            and row.precision >= target_precision
            and row.coverage >= min_coverage
            and row.n_alerts >= min_alerts
        ):
            return row
    return None


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pair counts.

    ``b``: items right under condition A and wrong under B; ``c``: the reverse.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
