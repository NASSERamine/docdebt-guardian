"""Run Auditor + Judge on the benchmark, with checkpoints and resume.

One row per item is appended to the output CSV every ``--checkpoint``
items; re-running the same command skips item_ids already in the file, so
an interrupted run (Ollama restart, API credit exhausted) resumes where it
stopped. Always run ``smoke_test.py`` first.

Example (Qwen, dev split, two voting patterns):
    python scripts/run_benchmark.py --model qwen2.5-coder:7b --patterns zeroshot cot \\
        --dataset /abs/path/verified_final_DEFINITIVE_363.xlsx \\
        --split-file /abs/path/config/split.csv --split dev \\
        --out /abs/path/runs/qwen_dev.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from docdebt.benchmark import detection_metrics, item_to_unit, load_items
from docdebt.core.auditor import Auditor
from docdebt.detection.judge import Judge, JudgeConfig, Status
from docdebt.llm.catalog import get_model_spec
from docdebt.llm.client import LLMClient
from docdebt.pipeline import build_registry
from docdebt.settings import load_settings

FIELDS = [
    "item_id",
    "label",
    "debt_type",
    "model",
    "patterns",
    "structural_debt",
    "structural_rules",
    "semantic_status",
    "reason",
    "confidence",
    "debt_votes",
    "clean_votes",
    "n_semantic_findings",
    "rejected_citations",
    "final_status",
    "votes_json",
    "explanation_raw",
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--patterns", nargs="+", default=["zeroshot"])
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N pending items")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the fixed item order")
    parser.add_argument("--checkpoint", type=int, default=20)
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Stop after this many consecutive items whose model calls failed",
    )
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

    # The file lists every debt item before every clean item. A fixed, seeded shuffle
    # mixes the classes so that --limit gives a representative subset, and the same
    # order is rebuilt on resume (the shuffle happens BEFORE done items are removed).
    ordered = items.to_dict("records")
    random.Random(args.seed).shuffle(ordered)
    already = done_ids(out)
    pending = [r for r in ordered if str(r["item_id"]) not in already]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(items)} item(s) in scope, {len(already)} already done, {len(pending)} to run -> {out}")

    settings = load_settings(args.config)
    client = LLMClient(get_model_spec(args.model, settings.model_specs()))
    judge = Judge(
        client,
        build_registry(settings),
        JudgeConfig(
            patterns=args.patterns,
            samples_per_pattern=args.samples,
            explain=not args.no_explain,
        ),
    )
    auditor = Auditor(settings.auditor.to_config())

    buffer: list[dict] = []
    consecutive_failures = 0
    for i, item in enumerate(pending, 1):
        started = time.perf_counter()
        unit = item_to_unit(str(item["item_id"]), item["method_code"], item["stale_doc"])
        row = {
            "item_id": item["item_id"],
            "label": item["label"],
            "debt_type": item["debt_type"],
            "model": args.model,
            "patterns": "+".join(args.patterns),
        }
        if unit is None:
            row.update(final_status="abstain", semantic_status="abstain", reason="unparsable_code")
        else:
            report = auditor.audit(unit)
            result = judge.judge(unit, report.findings)
            final = Status.DEBT if report.has_structural_debt else result.status
            row.update(
                structural_debt=report.has_structural_debt,
                structural_rules="|".join(f.rule for f in report.errors),
                semantic_status=result.status.value,
                reason=result.reason,
                confidence=round(result.confidence, 4),
                debt_votes=result.debt_votes,
                clean_votes=result.clean_votes,
                n_semantic_findings=len(result.semantic_findings),
                rejected_citations=result.rejected_citations,
                final_status=final.value,
                votes_json=json.dumps(
                    [
                        {
                            "pattern": v.pattern,
                            "sample": v.sample,
                            "verdict": v.verdict,
                            "raw": v.raw,
                            "version": v.prompt_version,
                            "truncated": v.truncated,
                            "error": v.error,
                        }
                        for v in result.votes
                    ],
                    ensure_ascii=False,
                ),
                explanation_raw=result.explanation_raw or "",
            )
        row["seconds"] = round(time.perf_counter() - started, 2)
        if unit is not None and any(v.error for v in result.votes):
            # A failed call is not a verdict: never write it, so a resumed run retries it.
            consecutive_failures += 1
            print(f"  !! {item['item_id']}: model call failed: {next(v.error for v in result.votes if v.error)}")
            if consecutive_failures >= args.max_failures:
                append_rows(out, buffer)
                raise SystemExit(
                    f"Stopped after {consecutive_failures} consecutive failures "
                    f"(is the model server up / is there API credit left?). "
                    f"Re-run the same command to resume."
                )
            continue
        consecutive_failures = 0
        buffer.append(row)
        if len(buffer) >= args.checkpoint:
            append_rows(out, buffer)
            buffer = []
            print(f"  checkpoint: {i}/{len(pending)} processed ({client.tracker.total_tokens} tokens so far)")
    if buffer:
        append_rows(out, buffer)

    report_metrics(out)


def report_metrics(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    for key, title in (("semantic_status", "LLM judge alone"), ("final_status", "Auditor + judge")):
        m = detection_metrics(rows, key)
        o = m["overall"]
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        print(
            f"\n{title} (n={o['n']}): coverage={fmt(o['coverage'])} "
            f"accuracy(answered)={fmt(o['accuracy_answered'])} recall={fmt(o['recall'])} "
            f"specificity={fmt(o['specificity'])} | recall structural={fmt(m.get('recall_structural'))} "
            f"semantic={fmt(m.get('recall_semantic'))}"
        )


if __name__ == "__main__":
    main()
