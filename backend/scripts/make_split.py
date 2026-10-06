"""Freeze a development / test split of the benchmark.

Stratified by debt type (structural, semantic, clean). Calibration items
(the few-shot examples) are excluded from both splits. The test split is
never used to tune prompts or thresholds; it only reports final numbers.

Example:
    python scripts/make_split.py --dataset /abs/path/data.xlsx \\
        --exclude-file /abs/path/config/fewshot_detection.json \\
        --dev-fraction 0.3 --seed 42 --out /abs/path/config/split.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from docdebt.benchmark import load_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="JSON from export_fewshot.py; its item_ids are excluded",
    )
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    excluded: set[str] = set()
    if args.exclude_file:
        excluded = {r["item_id"] for r in json.loads(args.exclude_file.read_text(encoding="utf-8"))}
    items = items[~items["item_id"].isin(excluded)]

    rng = random.Random(args.seed)
    rows = []
    for debt_type, group in sorted(items.groupby("debt_type"), key=lambda kv: kv[0]):
        ids = sorted(group["item_id"].astype(str))
        rng.shuffle(ids)
        n_dev = round(len(ids) * args.dev_fraction)
        rows += [(i, "dev", debt_type) for i in ids[:n_dev]]
        rows += [(i, "test", debt_type) for i in ids[n_dev:]]
        print(f"{debt_type:<11} dev={n_dev:<4} test={len(ids) - n_dev}")

    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.out.resolve().open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "split", "debt_type"])
        writer.writerows(sorted(rows))
    print(f"Excluded {len(excluded)} calibration item(s). Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
