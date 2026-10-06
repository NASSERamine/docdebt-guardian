"""Choose the Judge's confidence threshold from a benchmark run (dev split only).

Reads the CSV written by ``run_benchmark.py``, prints precision, recall,
specificity and coverage at every observed confidence, and writes the
chosen threshold to JSON. Items with structural debt are included as
they are judged by the LLM too; use ``--semantic-only`` to calibrate on
items without structural errors, where the LLM is the only signal.

Example:
    python scripts/calibrate.py --run /abs/path/runs/qwen_dev.csv --target-precision 0.85 \\
        --out /abs/path/config/threshold_qwen.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from docdebt.detection.calibration import CalibrationRecord, choose_threshold, threshold_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    parser.add_argument("--min-alerts", type=int, default=10)
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.run.resolve().open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.semantic_only:
        rows = [r for r in rows if r.get("structural_debt") != "True"]
    records = [
        CalibrationRecord(r["item_id"], r["label"] == "debt", r["semantic_status"], float(r["confidence"] or 0.0))
        for r in rows
        if r.get("semantic_status")
    ]
    fmt = lambda v: "  n/a " if v is None else f"{v:6.3f}"  # noqa: E731
    print(f"{len(records)} item(s)\n threshold coverage precision recall specificity alerts")
    for row in threshold_table(records):
        print(
            f"  {row.threshold:6.3f}  {fmt(row.coverage)}  {fmt(row.precision)}  {fmt(row.recall)}"
            f"  {fmt(row.specificity)}  {row.n_alerts:5d}"
        )

    chosen = choose_threshold(records, args.target_precision, args.min_coverage, args.min_alerts)
    if chosen is None:
        print(
            f"\nNo threshold reaches precision {args.target_precision}: do not use this "
            f"model alone for semantic alerts (escalate or abstain)."
        )
        return
    print(
        f"\nChosen threshold {chosen.threshold:.3f}: precision {chosen.precision:.3f}, "
        f"coverage {chosen.coverage:.3f}, recall {fmt(chosen.recall)}"
    )
    if args.out:
        args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.out.resolve().write_text(
            json.dumps(
                {
                    "run": str(args.run.resolve()),
                    "target_precision": args.target_precision,
                    "confidence_threshold": chosen.threshold,
                    "precision": chosen.precision,
                    "coverage": chosen.coverage,
                    "recall": chosen.recall,
                    "specificity": chosen.specificity,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
