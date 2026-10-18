"""Evaluate DocDebt Guardian on the research ground truth, with the paper's metrics,
and measure what retrieval brings (ablation A/B/C/D).

Every item runs through the application's own pipeline (``docdebt.service.engine``):
the exact rules, one detection call on the whole Javadoc, the field-by-field
repair (Replacement), with the model of the Model page (temperature 0). Four
context configurations are compared on the same items:

* ``A``: no context;
* ``B``: siblings only (documented methods of the same file, the research's best context);
* ``C``: hybrid search only (BM25 + all-MiniLM-L6-v2 vectors, RRF, top 4) in the
  repository's knowledge base;
* ``D``: siblings + hybrid search: the application's configuration.

Contexts need each item's repository at the commit before the fix:
``--repos-dir DIR`` with one folder per ground-truth method, ``DIR/<unique_id>/``,
holding the repository (or its Java files and README) at that commit. Items
without a folder run in A only. The knowledge base of an item is built from its
own folder, the item's method excluded (leak guard), exactly as the service does.

Output: one CSV row per (item, configuration) appended to ``--out`` (re-running
resumes), and a summary (JSON next to the CSV, and printed): detection accuracy,
recall (structural, semantic), specificity; structural resolution; semantic
outcomes (improved / unchanged / worsened / net); preservation; retrieval
Recall@4 and MRR of same-file methods; exact McNemar tests B vs D.

Measure on the development split while choosing anything, then ONCE on the
test split for the figures of the report.

Example:
    export OPENAI_API_KEY=sk-...
    python scripts/run_tool_benchmark.py --model gpt-5.6-luna --api-key-env OPENAI_API_KEY \\
        --price-in 1.25 --price-out 10 \\
        --dataset /abs/verified_final_DEFINITIVE_363.xlsx --split-file config/split.csv --split dev \\
        --repos-dir /abs/parent_commits --configs A B C D --out runs/tool_dev.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import tempfile
import time
from pathlib import Path

from docdebt.benchmark import cosine_distances, detection_metrics, item_to_unit, load_items
from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source
from docdebt.detection.calibration import mcnemar_exact
from docdebt.detection.context import README_NAMES
from docdebt.detection.judge import Status
from docdebt.evaluation import (
    mean,
    paired_mcnemar,
    preservation,
    recall_at_k,
    reciprocal_rank,
    semantic_outcome,
    semantic_summary,
)
from docdebt.llm.client import UsageTracker
from docdebt.rag.embeddings import choose_embedder
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.retriever import RagProvider
from docdebt.repair.repairer import RepairStatus
from docdebt.service.engine import build_service_deps
from docdebt.service.llm_settings import build_spec

FIELDS = [
    "item_id", "config", "label", "debt_type", "verdict", "rules", "context_items", "context_sources",
    "repair_status", "repair_reason", "repaired_doc", "structural_ok", "calls", "input_tokens",
    "output_tokens", "cost_usd", "seconds", "recall_at_4", "mrr",
]  # fmt: skip
OWNER, REPO = "bench", "item"


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {(row["item_id"], row["config"]) for row in csv.DictReader(fh)}


def append_row(path: Path, row: dict) -> None:
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_repo(folder: Path) -> tuple[dict[str, str], str | None]:
    files, readme = {}, None
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if rel.endswith(".java"):
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
        elif "/" not in rel and rel in README_NAMES:
            readme = path.read_text(encoding="utf-8", errors="replace")
    return files, readme


def locate_file(unit, files: dict[str, str], hint: str | None) -> str | None:
    """The file of the item's method: the ``file_path`` column, else the file declaring it."""
    if hint and hint in files:
        return hint
    for path, source in files.items():
        if any(m.name == unit.name and m.code.split("{", 1)[0] == unit.code.split("{", 1)[0]
               for m in parse_java_source(source, path)):  # fmt: skip
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the key")
    parser.add_argument("--price-in", type=float, default=None, help="US$ per million input tokens")
    parser.add_argument("--price-out", type=float, default=None, help="US$ per million output tokens")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--repos-dir", type=Path, default=None)
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--embeddings", choices=["minilm", "bm25"], default="minilm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        raise SystemExit(f"Set {args.api_key_env} to the API key of the provider of {args.model}.")
    spec = build_spec(args.model, key, price_in_per_m=args.price_in, price_out_per_m=args.price_out)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset)
    if args.split != "all":
        if args.split_file is None:
            raise SystemExit("--split dev/test needs --split-file (see make_split.py).")
        with args.split_file.resolve().open(newline="", encoding="utf-8") as fh:
            keep = {r["item_id"] for r in csv.DictReader(fh) if r["split"] == args.split}
        items = items[items["item_id"].astype(str).isin(keep)]
    rows = items.to_dict("records")
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    done = done_keys(out)
    embedder = choose_embedder(args.embeddings)
    auditor = Auditor()
    print(f"{len(rows)} item(s), configs {args.configs}, embeddings: {embedder.name if embedder else 'BM25 only'}")

    for n, item in enumerate(rows, start=1):
        item_id = str(item["item_id"])
        unit = item_to_unit(item_id, item["method_code"], item["stale_doc"])
        if unit is None:
            continue
        folder = args.repos_dir / str(item.get("unique_id", item_id.rsplit("_", 1)[0])) if args.repos_dir else None
        files, readme = read_repo(folder) if folder and folder.is_dir() else ({}, None)
        path = locate_file(unit, files, item.get("file_path")) if files else None
        if path is not None:
            unit = dataclasses.replace(unit, file_path=path)  # where the retrieval looks for siblings
        for config in args.configs:
            if (item_id, config) in done:
                continue
            if config != "A" and path is None:
                continue  # no repository at the parent commit for this item
            with tempfile.TemporaryDirectory() as tmp:
                kb = None
                if config in ("C", "D"):
                    kb = KnowledgeBase(Path(tmp) / "kb.sqlite3")
                    kb.rebuild(OWNER, REPO, files, readme=readme, embedder=embedder)
                tracker = UsageTracker()
                deps = build_service_deps(
                    spec, owner=OWNER, repo=REPO, tracker=tracker, knowledge=kb, embedder=embedder
                )
                rag: RagProvider = deps.rag
                if config == "A":
                    deps.router.context_provider = None
                    rag = None
                else:
                    if config == "C":
                        rag.max_siblings = 0
                    if config in ("B", "D"):
                        # The item's method, with its stale Javadoc, in its file at the parent commit.
                        rag.set_files({path: files[path]})
                    if config == "B":
                        rag.top_k = 0
                started = time.perf_counter()
                structural = auditor.audit(unit).findings
                routed = deps.router.route(unit, structural)
                context = rag.build(unit) if rag is not None else None
                findings = routed.findings
                verdict = (
                    "debt"
                    if routed.has_structural_debt or routed.semantic_status == Status.DEBT
                    else ("clean" if routed.semantic_status == Status.CLEAN else "abstain")
                )
                repaired, status, reason = None, "", ""
                if verdict == "debt":
                    result = deps.repairer.repair(unit, findings, context=context.text if context else None)
                    status, reason = result.status.value, result.reason
                    repaired = result.repaired if result.status == RepairStatus.VERIFIED else None
                checked = dataclasses.replace(unit, javadoc=repaired or unit.javadoc)
                structural_ok = not auditor.audit(checked).has_structural_debt
                r4 = mrr = None
                if config in ("C", "D") and context is not None and kb is not None:
                    same_file = {
                        f"{m.class_name}.{m.name}"
                        for m in parse_java_source(files[path], path)
                        if m.javadoc and m.name != unit.name
                    }
                    ranked = [i.label for i in rag.search(unit)]
                    r4, mrr = recall_at_k(ranked, same_file, 4), reciprocal_rank(ranked, same_file)
                append_row(
                    out,
                    {
                        "item_id": item_id,
                        "config": config,
                        "label": item["label"],
                        "debt_type": item["debt_type"],
                        "verdict": verdict,
                        "rules": ",".join(sorted({f.rule for f in findings if f.is_debt})),
                        "context_items": len(context.items) if context else 0,
                        "context_sources": ",".join(i.source for i in context.items) if context else "",
                        "repair_status": status,
                        "repair_reason": reason,
                        "repaired_doc": repaired or "",
                        "structural_ok": structural_ok,
                        "calls": tracker.calls,
                        "input_tokens": tracker.input_tokens,
                        "output_tokens": tracker.output_tokens,
                        "cost_usd": round(tracker.cost_usd, 6),
                        "seconds": round(time.perf_counter() - started, 2),
                        "recall_at_4": r4,
                        "mrr": mrr,
                    },
                )
        if n % 10 == 0:
            print(f"  {n}/{len(rows)} items")

    summarize(out, items)


def summarize(out: Path, items) -> None:
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_item = {str(r["item_id"]): r for r in items.to_dict("records")}
    summary: dict[str, dict] = {}
    correct: dict[str, dict[str, bool]] = {}
    resolved: dict[str, dict[str, bool]] = {}
    for config in sorted({r["config"] for r in rows}):
        part = [r for r in rows if r["config"] == config]
        det = detection_metrics([{**r, "status": r["verdict"]} for r in part], "status")
        debt = [r for r in part if r["label"] == "debt"]
        semantic = [r for r in debt if r["debt_type"] == "semantic"]
        refs, stales, repaireds = [], [], []
        for r in semantic:
            truth = by_item[r["item_id"]]
            refs.append(str(truth["javadoc_after"]))
            stales.append(str(truth["javadoc_before"]))
            repaireds.append(r["repaired_doc"] or str(truth["javadoc_before"]))
        outcomes = []
        if semantic:
            d_stale = cosine_distances(stales, refs)
            d_rep = cosine_distances(repaireds, refs)
            outcomes = [
                semantic_outcome(s, rep if r["repaired_doc"] else None)
                for s, rep, r in zip(d_stale, d_rep, semantic, strict=True)
            ]
        kept = total = 0
        for r in debt:
            truth = by_item[r["item_id"]]
            before, after = str(truth["javadoc_before"]), str(truth["javadoc_after"])
            k, t = preservation(before, after, r["repaired_doc"] or None)
            kept, total = kept + k, total + t
        summary[config] = {
            "items": len(part),
            "detection": det,
            "structural_resolution": mean(
                [1.0 if r["structural_ok"] == "True" else 0.0 for r in debt if r["debt_type"] == "structural"]
            ),
            "semantic": semantic_summary(outcomes),
            "preservation": kept / total if total else None,
            "retrieval_recall_at_4": mean([float(r["recall_at_4"]) for r in part if r["recall_at_4"]]),
            "retrieval_mrr": mean([float(r["mrr"]) for r in part if r["mrr"]]),
            "calls_per_item": mean([float(r["calls"]) for r in part]),
            "cost_usd": sum(float(r["cost_usd"] or 0) for r in part),
        }
        correct[config] = {r["item_id"]: (r["verdict"] == "debt") == (r["label"] == "debt") for r in part}
        resolved[config] = {r["item_id"]: r["structural_ok"] == "True" for r in debt}
    if "B" in correct and "D" in correct:
        summary["mcnemar_B_vs_D"] = {
            "detection": paired_mcnemar(correct["B"], correct["D"], mcnemar_exact),
            "structural_resolution": paired_mcnemar(resolved["B"], resolved["D"], mcnemar_exact),
        }
    path = out.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary written to {path}")


if __name__ == "__main__":
    main()
