"""Confusion matrix and detection metrics of a detection run CSV.

Recomputes a published table from the run's own rows (no model call): the
items of the run, the label of each one and the status it received. Used for
the raw-verdict measure of the prototype (Qwen2.5-Coder 7B, zero-shot).

Example:
    python scripts/summarize_detection_run.py runs/qwen_zeroshot_noexplain.csv
    python scripts/summarize_detection_run.py runs/x.csv --status final_status
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--status", default="semantic_status", help="column holding debt / clean / abstain")
    args = parser.parse_args()

    with args.run.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    counts = Counter((r["label"], r[args.status]) for r in rows)
    models = sorted({r.get("model", "") for r in rows})
    tp, fn = counts[("debt", "debt")], counts[("debt", "clean")]
    fp, tn = counts[("clean", "debt")], counts[("clean", "clean")]
    abstain = sum(v for (_, s), v in counts.items() if s not in ("debt", "clean"))
    n_debt, n_clean = tp + fn + counts[("debt", "abstain")], fp + tn + counts[("clean", "abstain")]
    answered = tp + fn + fp + tn

    print(f"Run        : {args.run} ({len(rows)} items, model(s): {', '.join(models)}, column {args.status})")
    print(f"{'':22}{'predicted debt':>16}{'predicted clean':>17}")
    print(f"{f'actual debt ({n_debt})':22}{tp:>16}{fn:>17}")
    print(f"{f'actual clean ({n_clean})':22}{fp:>16}{tn:>17}")
    print(f"Abstentions: {abstain}")
    print(f"Recall     : {tp / n_debt:.3f}" if n_debt else "Recall     : -")
    print(f"Specificity: {tn / n_clean:.3f}" if n_clean else "Specificity: -")
    print(f"Accuracy   : {(tp + tn) / answered:.3f} (answered items)" if answered else "Accuracy   : -")


if __name__ == "__main__":
    main()
