"""Export repair few-shot examples (one-shot and three-shot patterns) to JSON.

Each example is a benchmark method: its stale Javadoc (``javadoc_before``) and
the developer's fix (``javadoc_after``). You state explicitly which examples
each pattern uses; nothing is guessed. Zero-shot and CoT need no file.

Example (ids taken from the research notebook, in research order):
    python scripts/export_repair_fewshot.py --dataset datasets/verified_final_DEFINITIVE_363.xlsx \\
        --unique-ids 8126 4308 1489 --oneshot 0 --threeshot 0 1 2 \\
        --out config/fewshot_repair.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docdebt.benchmark import load_items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--unique-ids", nargs="+", required=True, help="unique_id of each example, in order")
    parser.add_argument("--oneshot", nargs="+", type=int, required=True, help="positions used by one-shot")
    parser.add_argument("--threeshot", nargs="+", type=int, required=True, help="positions used by three-shot")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    debt = items[items["label"] == "debt"].set_index("item_id")
    examples = []
    for unique_id in args.unique_ids:
        item_id = f"{unique_id}_debt"
        if item_id not in debt.index:
            raise SystemExit(f"Unknown debt item: {item_id}")
        row = debt.loc[item_id]
        examples.append(
            {
                "item_id": item_id,
                "method_code": row["method_code"],
                "stale_doc": row["stale_doc"],
                "repaired_doc": row["javadoc_after"],
            }
        )
    payload = {"examples": examples, "oneshot": args.oneshot, "threeshot": args.threeshot}
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for i, ex in enumerate(examples):
        print(f"[{i}] {ex['item_id']}")
    print(f"one-shot uses {args.oneshot}, three-shot uses {args.threeshot}\nWrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
