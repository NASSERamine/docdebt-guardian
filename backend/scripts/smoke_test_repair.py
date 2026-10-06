"""Smoke test of the repair step: print the RAW model answers and what the tool does with them.

For every method it shows the edit plan, then for each attempt the raw answer,
the block extracted from it, the Verifier's issues, and finally the diff of the
verified repair. Run it whenever the model, the pattern or a budget changes.

Examples:
    python scripts/smoke_test_repair.py --model qwen2.5-coder:7b
    python scripts/smoke_test_repair.py --model qwen2.5-coder:7b \\
        --dataset datasets/verified_final_DEFINITIVE_363.xlsx --n 3
    python scripts/smoke_test_repair.py --model gpt-5.6 --wording 0   # GPT-5.6 is not screened
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source
from docdebt.pipeline import build_repairer
from docdebt.repair.apply import javadoc_diff
from docdebt.settings import load_settings

BUILTIN = """class Demo {
    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) { return tempInKelvin - 273.15; }

    /**
     * Reads a file.
     * @param path the path
     * @return the content
     */
    public String read(String path) throws java.io.IOException { return path; }

    /**
     * Sums two numbers.
     * @param a first value
     * @return the sum
     */
    public int add(int a, int b) { return a + b; }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--pattern", default="zeroshot")
    parser.add_argument("--wording", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None, help="Benchmark file (else built-in examples)")
    parser.add_argument("--n", type=int, default=3, help="Structural-debt items to draw with --dataset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings(args.config)
    repairer = build_repairer(settings, args.model, pattern=args.pattern, wording=args.wording)
    spec = repairer.client.spec
    print(
        f"Model: {spec.id} ({spec.provider}), repair budget: {spec.max_tokens.get('repair')}, "
        f"pattern: {args.pattern}, attempts: {repairer.config.max_attempts}"
    )

    if args.dataset:
        from docdebt.benchmark import item_to_unit, load_items

        items = load_items(args.dataset)
        pool = items[(items["label"] == "debt") & (items["debt_type"] == "structural")].to_dict("records")
        picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
        cases = [(r["item_id"], item_to_unit(r["item_id"], r["method_code"], r["stale_doc"])) for r in picked]
    else:
        cases = [(u.name, u) for u in parse_java_source(BUILTIN, "Demo.java")]

    auditor = Auditor(settings.auditor.to_config())
    for name, unit in cases:
        print("=" * 100)
        if unit is None:
            print(f"{name}: method code did not parse; skipped")
            continue
        findings = auditor.audit(unit).findings
        result = repairer.repair(unit, findings)
        print(
            f"{name}: plan add={sorted(r.label for r in result.plan.add)} "
            f"remove={sorted(r.label for r in result.plan.remove)} "
            f"replace={sorted(r.label for r in result.plan.replace)}"
        )
        for attempt in result.attempts:
            print(f"  --- attempt {attempt.number}: truncated={attempt.truncated} error={attempt.error}")
            print("  RAW: " + (attempt.raw or "<empty>").replace("\n", "\n       "))
            print(f"  extracted block: {'yes' if attempt.block else 'NO'}")
            for issue in attempt.issues:
                print(f"  issue [{issue.code}] {issue.message}")
        print(f"  => status={result.status.value} reason={result.reason or '-'} prompt={result.prompt_version}")
        if result.repaired:
            print(javadoc_diff(unit, result))
    t = repairer.client.tracker
    print("=" * 100)
    print(f"{t.calls} call(s), {t.input_tokens} input + {t.output_tokens} output tokens")


if __name__ == "__main__":
    main()
