"""Benchmark helpers: load the research dataset, turn items into methods, compute metrics.

Requires the ``bench`` extra (pandas, openpyxl).

Item conventions (same as the research): every ground-truth method gives a
debt item (``<unique_id>_debt``, Javadoc before the fix) and a clean item
(``<unique_id>_clean``, Javadoc after the fix). The debt type of a debt item
comes from its severity: ``critical_signature_change`` is structural,
``substantial_semantic_change`` is semantic.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from docdebt.core.java_parser import parse_method_snippet
from docdebt.core.models import MethodUnit

REQUIRED_PAIRED = ("item_id", "method_code", "stale_doc", "label", "severity")
SEVERITY_TO_TYPE = {
    "critical_signature_change": "structural",
    "substantial_semantic_change": "semantic",
}


def build_paired_items(ground_truth: Any) -> Any:
    """Research ``build_paired_dataset``: one debt and one clean item per method."""
    import pandas as pd

    debt = ground_truth.copy()
    debt["item_id"] = debt["unique_id"].astype(str) + "_debt"
    debt["stale_doc"] = debt["javadoc_before"]
    debt["label"] = "debt"
    clean = ground_truth.copy()
    clean["item_id"] = clean["unique_id"].astype(str) + "_clean"
    clean["stale_doc"] = clean["javadoc_after"]
    clean["label"] = "clean"
    paired = pd.concat([debt, clean], ignore_index=True)
    if not paired["item_id"].is_unique:
        raise ValueError("item_id collision: unique_id is not unique in the ground truth.")
    return paired


def load_items(path: str | Path) -> Any:
    """Load paired items from an .xlsx/.csv file (paired or ground-truth layout)."""
    import pandas as pd

    path = Path(path).resolve()
    frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    if "item_id" not in frame.columns and {"javadoc_before", "javadoc_after"} <= set(frame.columns):
        frame = build_paired_items(frame)
    missing = [c for c in REQUIRED_PAIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(frame.columns)}.")
    frame = frame.copy()
    frame["debt_type"] = [
        "clean" if label == "clean" else SEVERITY_TO_TYPE.get(str(sev), str(sev))
        for label, sev in zip(frame["label"], frame["severity"], strict=True)
    ]
    frame["stale_doc"] = frame["stale_doc"].fillna("").astype(str)
    return frame


def item_to_unit(item_id: str, method_code: str, stale_doc: str) -> MethodUnit | None:
    """MethodUnit for a benchmark item: the item's code with the item's Javadoc attached."""
    unit = parse_method_snippet(str(method_code), file_path=str(item_id))
    if unit is None or not unit.syntax_ok:
        return None
    return dataclasses.replace(unit, javadoc=stale_doc or None, javadoc_start_line=None)


def detection_metrics(rows: list[dict[str, Any]], status_key: str) -> dict[str, Any]:
    """Accuracy, recall, specificity, coverage for one status column.

    ``rows`` need ``label`` (debt/clean), ``debt_type`` and ``status_key``
    (debt/clean/abstain). Abstentions are excluded from accuracy and counted
    in coverage; recall and specificity use all items of their class, so an
    abstention on a debt item lowers recall.
    """

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(subset)
        answered = [r for r in subset if r[status_key] in ("debt", "clean")]
        debt = [r for r in subset if r["label"] == "debt"]
        clean = [r for r in subset if r["label"] == "clean"]
        correct = sum((r[status_key] == "debt") == (r["label"] == "debt") for r in answered)
        return {
            "n": n,
            "coverage": len(answered) / n if n else None,
            "accuracy_answered": correct / len(answered) if answered else None,
            "recall": sum(r[status_key] == "debt" for r in debt) / len(debt) if debt else None,
            "specificity": sum(r[status_key] == "clean" for r in clean) / len(clean) if clean else None,
        }

    out = {"overall": block(rows)}
    for debt_type in ("structural", "semantic"):
        subset = [r for r in rows if r["debt_type"] == debt_type]
        if subset:
            out[f"recall_{debt_type}"] = block(subset)["recall"]
    return out


def cosine_distances(left: list[str], right: list[str], embedder: Any = None) -> list[float]:
    """Cosine distance between paired texts (all-MiniLM-L6-v2 embeddings by default).

    Texts are compared as plain prose (comment markers removed, whitespace
    collapsed). ``embedder`` is any object with ``encode(list, show_progress_bar=False)``;
    it defaults to ``SentenceTransformer("all-MiniLM-L6-v2")`` (needs the ``context`` extra).
    """
    import numpy as np

    from docdebt.core.javadoc import javadoc_plain_text

    if embedder is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    a = np.asarray(embedder.encode([javadoc_plain_text(t) for t in left], show_progress_bar=False))
    b = np.asarray(embedder.encode([javadoc_plain_text(t) for t in right], show_progress_bar=False))
    sims = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return [float(1.0 - x) for x in sims]
