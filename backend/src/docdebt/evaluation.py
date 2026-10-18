"""Evaluation of the tool on the research ground truth, with the metrics of the paper.

Detection (paired debt and clean items): accuracy, recall (overall, structural,
semantic) and specificity, computed by :func:`docdebt.benchmark.detection_metrics`.

Repair (debt items only), against ``javadoc_after`` (the developer's fix):

* **structural resolution**: the repaired Javadoc passes every exact rule
  (each real parameter, the return of a non-void method and each declared
  exception documented, no phantom parameter);
* **semantic outcome** (semantic items): ``improved`` when the repaired
  Javadoc is closer to the reference than the stale one it started from,
  ``worsened`` when farther, ``unchanged`` otherwise (same distance, or no
  repair); the **net** change is improved minus worsened, over all items.
  Distances are cosine distances of all-MiniLM-L6-v2 embeddings of the
  comment text without markers, as in the paper;
* **preservation**: among the fields the developer left unchanged, the share
  the repair also left unchanged.

Retrieval: for the hybrid search, Recall@k and MRR of the documented methods
of the item's own file (the context the research found most useful): a good
search finds them even without being told which file the method is in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from docdebt.core.javadoc_edit import field_text, normalize_text, parse_layout


def field_texts(doc: str | None) -> dict[str, str]:
    """``{"description": ..., "@param x": ..., "@return": ...}`` of a Javadoc (normalised text)."""
    if not doc or not doc.strip():
        return {}
    out: dict[str, str] = {}
    for block in parse_layout(doc).blocks:
        if block.ref is not None:
            out[block.ref.label] = normalize_text(field_text(block))
    return out


def preservation(before: str | None, after: str | None, repaired: str | None) -> tuple[int, int]:
    """``(kept, total)``: fields the developer left unchanged, and how many the repair kept too.

    A missing repair counts as keeping everything (the stale Javadoc stays).
    """
    b, a = field_texts(before), field_texts(after)
    r = field_texts(repaired) if repaired else b
    unchanged = [k for k in b if k in a and a[k] == b[k]]
    return sum(1 for k in unchanged if r.get(k) == b[k]), len(unchanged)


def semantic_outcome(distance_stale: float, distance_repaired: float | None, tolerance: float = 1e-6) -> str:
    """``improved`` / ``unchanged`` / ``worsened`` of one repair against the reference."""
    if distance_repaired is None or abs(distance_repaired - distance_stale) <= tolerance:
        return "unchanged"
    return "improved" if distance_repaired < distance_stale else "worsened"


def semantic_summary(outcomes: Iterable[str]) -> dict[str, float | int | None]:
    outcomes = list(outcomes)
    n = len(outcomes)
    if not n:
        return {"n": 0, "improved": None, "unchanged": None, "worsened": None, "net": None}
    share = {k: outcomes.count(k) / n for k in ("improved", "unchanged", "worsened")}
    return {"n": n, **share, "net": share["improved"] - share["worsened"]}


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float | None:
    if not relevant:
        return None
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def paired_mcnemar(
    rows_a: dict[str, bool], rows_b: dict[str, bool], test: Callable[[int, int], float]
) -> dict[str, float | int]:
    """Exact McNemar test between two configurations on the items both ran."""
    common = sorted(set(rows_a) & set(rows_b))
    b = sum(1 for i in common if rows_a[i] and not rows_b[i])
    c = sum(1 for i in common if rows_b[i] and not rows_a[i])
    return {"n": len(common), "a_only": b, "b_only": c, "p": test(b, c)}
