"""Outcome of a repair run CSV: statuses, attempts, failures and mean duration.

Recomputes a published table from the run's own rows (no model call), for
example the structural repair of the prototype (Qwen2.5-Coder 7B, regeneration).

Example:
    python scripts/summarize_repair_run.py runs/repair_qwen_structural.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()

    with args.run.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    status = Counter(r["status"] for r in rows)
    to_repair = [r for r in rows if r["status"] != "not_needed"]
    verified = [r for r in to_repair if r["status"] == "verified"]
    first = sum(1 for r in verified if r["attempts"] == "1")
    seconds = [float(r["seconds"]) for r in rows if r.get("seconds")]
    models = sorted({r.get("model", "") for r in rows})

    print(f"Run        : {args.run} ({len(rows)} items, model(s): {', '.join(models)})")
    print(f"Statuses   : {dict(status)}")
    print(f"Not needed : {status['not_needed']}")
    print(f"To repair  : {len(to_repair)}")
    share = f" ({len(verified) / len(to_repair):.0%})" if to_repair else ""
    print(f"Verified   : {len(verified)}/{len(to_repair)}{share}")
    if verified:
        print(f"  attempt 1: {first} ({first / len(verified):.0%})")
        print(f"  attempt 2+: {len(verified) - first} ({(len(verified) - first) / len(verified):.0%})")
    print(f"Failures   : {len(to_repair) - len(verified)}")
    if seconds:
        print(f"Mean time  : {sum(seconds) / len(seconds):.1f} s per item")


if __name__ == "__main__":
    main()
