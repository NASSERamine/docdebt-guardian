"""Smoke test: print the RAW answers of one model before any long run.

Run this every time a model, a prompt pattern or a budget changes. Read the
raw output yourself: a verdict word missing, a truncated chain of thought or
a citation block in the wrong format is visible here in seconds, and
costs nothing compared with a failed 720-item run.

Examples:
    python scripts/smoke_test.py --model qwen2.5-coder:7b
    python scripts/smoke_test.py --model qwen2.5-coder:7b --patterns zeroshot cot \\
        --dataset /abs/path/verified_final_DEFINITIVE_363.xlsx --n 3
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source
from docdebt.detection.judge import Judge, JudgeConfig
from docdebt.llm.catalog import get_model_spec
from docdebt.llm.client import LLMClient
from docdebt.pipeline import build_registry
from docdebt.settings import load_settings

BUILTIN = """class Demo {
    /**
     * Returns the maximum of the two values.
     * @param a first value
     * @param b second value
     * @return the larger value
     */
    public int pick(int a, int b) { return Math.min(a, b); }

    /**
     * Returns the sum of the two values.
     * @param a first value
     * @param b second value
     * @return a + b
     */
    public int add(int a, int b) { return a + b; }

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) { return tempInKelvin - 273.15; }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--patterns", nargs="+", default=["zeroshot", "cot"])
    parser.add_argument("--config", type=Path, default=None, help="Optional .docdebt.yml")
    parser.add_argument("--dataset", type=Path, default=None, help="Benchmark file (else built-in examples)")
    parser.add_argument("--n", type=int, default=3, help="Items per class when --dataset is given")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings(args.config)
    spec = get_model_spec(args.model, settings.model_specs())
    client = LLMClient(spec)
    registry = build_registry(settings)
    judge = Judge(client, registry, JudgeConfig(patterns=args.patterns))

    if args.dataset:
        from docdebt.benchmark import item_to_unit, load_items

        items = load_items(args.dataset)
        rng = random.Random(args.seed)
        picked = []
        for debt_type in ("structural", "semantic", "clean"):
            pool = items[items["debt_type"] == debt_type].to_dict("records")
            picked += rng.sample(pool, min(args.n, len(pool)))
        cases = [
            (
                r["item_id"],
                r["debt_type"],
                item_to_unit(r["item_id"], r["method_code"], r["stale_doc"]),
            )
            for r in picked
        ]
    else:
        cases = [(u.name, "builtin", u) for u in parse_java_source(BUILTIN, "Demo.java")]

    print(f"Model: {spec.id} ({spec.provider}), budgets: {spec.max_tokens}, num_ctx: {spec.num_ctx}")
    for item_id, debt_type, unit in cases:
        print("=" * 100)
        print(f"{item_id}  [{debt_type}]")
        if unit is None:
            print("  !! method code did not parse; skipped")
            continue
        structural = Auditor().audit(unit).findings
        print(f"  Auditor: {[f.rule for f in structural if f.is_debt] or 'no structural error'}")
        result = judge.judge(unit, structural)
        for vote in result.votes:
            print(
                f"  --- {vote.pattern} #{vote.sample}  version={vote.prompt_version}  "
                f"truncated={vote.truncated}  parsed={vote.verdict}"
            )
            print("  RAW: " + (vote.raw or f"<error: {vote.error}>").replace("\n", "\n       "))
        if result.explanation_raw is not None:
            print("  --- citation prompt RAW:")
            print("       " + result.explanation_raw.replace("\n", "\n       "))
            print(f"  rejected citations: {result.rejected_citations}")
        print(f"  => status={result.status.value} reason={result.reason} confidence={result.confidence:.2f}")
    print("=" * 100)
    t = client.tracker
    print(f"{t.calls} call(s), {t.input_tokens} input + {t.output_tokens} output tokens")


if __name__ == "__main__":
    main()
