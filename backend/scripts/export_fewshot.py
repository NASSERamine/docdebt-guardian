"""Export the six few-shot calibration items, in their research order, to JSON.

The six item_id values must be given explicitly, in the order used in the
research (one-shot uses positions 0 and 4, three-shot 0, 2 and 4). They are
never guessed: take them from the research notebook output.

Example:
    python scripts/export_fewshot.py --dataset /abs/path/verified_final_DEFINITIVE_363.xlsx \\
        --item-ids ID0 ID1 ID2 ID3 ID4 ID5 --out /abs/path/config/fewshot_detection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docdebt.benchmark import load_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--item-ids", nargs=6, required=True, metavar="ITEM_ID")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset).set_index("item_id")
    unknown = [i for i in args.item_ids if i not in items.index]
    if unknown:
        raise SystemExit(f"Unknown item_id(s): {unknown}")

    rows = []
    for item_id in args.item_ids:
        item = items.loc[item_id]
        rows.append(
            {
                "item_id": item_id,
                "method_code": item["method_code"],
                "stale_doc": item["stale_doc"],
                "correct_verdict": "INCOMPLETE" if item["label"] == "debt" else "COMPLETE",
                "debt_type": item["debt_type"],
            }
        )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for i, row in enumerate(rows):
        print(f"[{i}] {row['item_id']:<24} {row['correct_verdict']:<10} {row['debt_type']}")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
