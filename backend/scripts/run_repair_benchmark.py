"""Run the repair step on the benchmark, with checkpoints and resume.

Two modes:

* ``structural`` (default): the structural-debt items only, with the Auditor's
  findings as input. Nothing else is involved, so this measures the Repairer
  and the Verifier alone. Every verified repair passes the Auditor by
  construction; the numbers to read are how often a verified repair is found,
  how many attempts it takes, and why the others fail.
* ``pipeline``: every item of the split goes through detection first (Auditor
  and Judge, as in production) and is repaired when flagged. On debt items it
  shows how many are detected and repaired; on clean items, how many are
  modified by mistake (preservation).

``--similarity`` adds the cosine distance between the final Javadoc and the
developer's fix (research threshold 0.15), and the same distance for the
untouched stale Javadoc as a baseline (needs sentence-transformers).

Always run ``smoke_test_repair.py`` first. Re-running the same command resumes.

Example:
    python scripts/run_repair_benchmark.py --model qwen2.5-coder:7b --limit 50 \\
        --dataset datasets/verified_final_DEFINITIVE_363.xlsx \\
        --split-file config/split.csv --split dev --out runs/repair_qwen_structural.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections import Counter
from pathlib import Path

from docdebt.benchmark import cosine_distances, item_to_unit, load_items
from docdebt.core.auditor import Auditor
from docdebt.pipeline import build_detection_pipeline, build_repairer
from docdebt.repair.repairer import RepairStatus
from docdebt.settings import load_settings

FIELDS = [
    "item_id",
    "label",
    "debt_type",
    "mode",
    "model",
    "pattern",
    "detected",
    "plan",
    "status",
    "reason",
    "attempts",
    "changes",
    "unresolved",
    "repaired_doc",
    "seconds",
]


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["item_id"] for row in csv.DictReader(fh)}


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Repair model")
    parser.add_argument("--judge-model", default=None, help="Detection model in pipeline mode (default: --model)")
    parser.add_argument("--pattern", default="zeroshot")
    parser.add_argument("--wording", type=int, default=None)
    parser.add_argument("--mode", choices=["structural", "pipeline"], default="structural")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=int, default=20)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--similarity", action="store_true")
    parser.add_argument("--hint-findings", action="store_true", help="Give the repair model the findings")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset)
    if args.split != "all":
        if args.split_file is None:
            raise SystemExit("--split dev/test needs --split-file (see make_split.py).")
        with args.split_file.resolve().open(newline="", encoding="utf-8") as fh:
            keep = {r["item_id"] for r in csv.DictReader(fh) if r["split"] == args.split}
        items = items[items["item_id"].astype(str).isin(keep)]
    if args.mode == "structural":
        items = items[(items["label"] == "debt") & (items["debt_type"] == "structural")]

    ordered = items.to_dict("records")
    random.Random(args.seed).shuffle(ordered)
    already = done_ids(out)
    pending = [r for r in ordered if str(r["item_id"]) not in already]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(items)} item(s) in scope, {len(already)} already done, {len(pending)} to run -> {out}")

    settings = load_settings(args.config)
    repairer = build_repairer(
        settings,
        args.model,
        pattern=args.pattern,
        wording=args.wording,
        hint_findings=True if args.hint_findings else None,
    )
    auditor = Auditor(settings.auditor.to_config())
    detector = None
    if args.mode == "pipeline":
        detector = build_detection_pipeline(settings, cascade=[args.judge_model or args.model])

    buffer: list[dict] = []
    failures = 0
    for i, item in enumerate(pending, 1):
        started = time.perf_counter()
        unit = item_to_unit(str(item["item_id"]), item["method_code"], item["stale_doc"])
        row = {
            "item_id": item["item_id"],
            "label": item["label"],
            "debt_type": item["debt_type"],
            "mode": args.mode,
            "model": args.model,
            "pattern": args.pattern,
        }
        if unit is None:
            row.update(status="skipped", reason="unparsable_code", detected="")
        else:
            if detector is not None:
                routed = detector.run_one(unit)
                findings = routed.findings if routed.status.value == "debt" else []
                row["detected"] = routed.status.value
            else:
                findings = auditor.audit(unit).findings
                row["detected"] = "structural"
            result = repairer.repair(unit, findings)
            if result.llm_failed:
                failures += 1
                print(f"  !! {item['item_id']}: model call failed: {result.attempts[0].error}")
                if failures >= args.max_failures:
                    append_rows(out, buffer)
                    raise SystemExit(f"Stopped after {failures} consecutive failures. Re-run to resume.")
                continue
            failures = 0
            row.update(
                plan=";".join(sorted(r.label for r in result.plan.touched)),
                status=result.status.value,
                reason=result.reason,
                attempts=len(result.attempts),
                changes=";".join(f"{c.action}:{c.ref.label}" for c in result.changes),
                unresolved=";".join(result.unresolved),
                repaired_doc=result.repaired or "",
            )
        row["seconds"] = round(time.perf_counter() - started, 2)
        buffer.append(row)
        if len(buffer) >= args.checkpoint:
            append_rows(out, buffer)
            buffer = []
            print(
                f"  checkpoint: {i}/{len(pending)} processed "
                f"({repairer.client.tracker.total_tokens} tokens so far)"
            )
    append_rows(out, buffer)
    report(out, items, args)


def report(path: Path, items, args) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    print(f"\n{len(rows)} item(s) in {path.name}")
    by_status = Counter(r["status"] for r in rows if r["label"] == "debt")
    debt = [r for r in rows if r["label"] == "debt"]
    print("Debt items by repair status: " + ", ".join(f"{n} {s}" for s, n in by_status.most_common()))
    if debt:
        verified = [r for r in debt if r["status"] == RepairStatus.VERIFIED.value]
        flagged = [r for r in debt if r["status"] not in ("not_needed", "skipped")]
        mean_attempts = sum(int(r["attempts"]) for r in verified) / max(len(verified), 1)
        share = f" ({len(verified) / len(flagged):.1%})" if flagged else ""
        print(
            f"  verified among flagged debt (repair quality): {len(verified)}/{len(flagged)}{share}; "
            f"mean attempts on verified: {mean_attempts:.2f}"
        )
        print(
            f"  verified among all debt items (includes {len(debt) - len(flagged)} the detector "
            f"did not flag): {len(verified)}/{len(debt)} ({len(verified) / len(debt):.1%})"
        )
        reasons = Counter(r["reason"] for r in debt if r["status"] == RepairStatus.FAILED.value)
        for reason, n in reasons.most_common():
            print(f"  failed because: {n:4d}  {reason}")
    if args.mode == "pipeline":
        detected = [r for r in debt if r["detected"] == "debt"]
        print(f"  detected as debt: {len(detected)}/{len(debt)}")
        clean = [r for r in rows if r["label"] == "clean"]
        if clean:
            flagged = [r for r in clean if r["detected"] == "debt"]
            changed = [r for r in clean if r["status"] == RepairStatus.VERIFIED.value]
            print(
                f"Clean items: {len(flagged)}/{len(clean)} flagged, {len(changed)}/{len(clean)} modified "
                f"(preservation {1 - len(changed) / len(clean):.1%})"
            )
    if args.similarity and debt:
        after = items.set_index(items["item_id"].astype(str))["javadoc_after"].to_dict()
        stale = items.set_index(items["item_id"].astype(str))["stale_doc"].to_dict()
        final = [r["repaired_doc"] if r["status"] == "verified" else stale[r["item_id"]] for r in debt]
        refs = [after[r["item_id"]] for r in debt]
        after_repair = cosine_distances(final, refs)
        untouched = cosine_distances([stale[r["item_id"]] for r in debt], refs)
        n = len(debt)
        print(
            f"Cosine distance to the developer's fix <= 0.15: repaired pipeline "
            f"{sum(d <= 0.15 for d in after_repair) / n:.1%} vs untouched stale Javadoc "
            f"{sum(d <= 0.15 for d in untouched) / n:.1%}"
        )


if __name__ == "__main__":
    main()
