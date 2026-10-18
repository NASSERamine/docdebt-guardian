"""Check the application's model end to end, with a real API key (increment 2).

Two steps, exactly as the service does them:

1. the connection test of the Model page (one short prompt; a refused key
   stops here, as it would save nothing in the application);
2. one detection per method of a small built-in Java class, with the
   application's own pipeline (``docdebt.service.engine``): exact rules OR the
   model (one zero-shot call, temperature 0), then the model's explanation.

The three methods cover the three cases of the unified detection:

* ``toCelsius``: a phantom ``@param`` -> debt found by an exact rule;
* ``pick``: the description says "maximum", the code returns the minimum ->
  debt only the model can find, with its explanation;
* ``add``: correct documentation -> no debt.

Prints, for each method, the expected and obtained verdict, the rules, the
model's explanation (quotes found word for word are marked), the prompt
version and the cost; exits with 1 if any verdict differs from the expected
one (a difference of the model is a result to discuss, not a crash).

Example:
    export OPENAI_API_KEY=sk-...
    python scripts/check_model.py --model gpt-5.6-luna --price-in 1.25 --price-out 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import warnings
from pathlib import Path

from docdebt.core.java_parser import parse_java_source
from docdebt.detection.judge import Status
from docdebt.llm.catalog import PROVIDER_LABELS, SERVICE_MODELS, get_model_spec
from docdebt.llm.client import UsageTracker
from docdebt.llm.prompts.registry import is_screened
from docdebt.repair.repairer import RepairStatus
from docdebt.service import engine
from docdebt.service.engine import build_regenerator, build_service_deps
from docdebt.service.llm_settings import build_spec, test_connection

SOURCE = """package demo;

public class Temperature {

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) {
        return tempInKelvin - 273.15;
    }

    /**
     * Returns the maximum of the two values.
     * @param a first value
     * @param b second value
     * @return the larger of a and b
     */
    public int pick(int a, int b) {
        return Math.min(a, b);
    }

    /**
     * Returns the sum of the two values.
     * @param a first value
     * @param b second value
     * @return the sum of a and b
     */
    public int add(int a, int b) {
        return a + b;
    }
}
"""

EXPECTED = {
    "toCelsius": ("debt", "exact rule (phantom @param)"),
    "pick": ("debt", "model only (description contradicted by the code)"),
    "add": ("clean", "correct documentation"),
}

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}


def verdict_of(routed) -> str:
    if routed.status == Status.DEBT:
        return "debt"
    return "clean" if routed.status == Status.CLEAN else "no answer"


def repair_both(deps, spec, tracker, unit, findings) -> list[dict]:
    """The proposal's fix (replacement) and the on-demand regeneration, each re-checked by the exact rules."""
    out = []
    regenerator = build_regenerator(spec, tracker=tracker)
    for mechanism, repairer in (("replacement", deps.repairer), ("regeneration", regenerator)):
        before = (tracker.calls, tracker.cost_usd)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unscreened regeneration wording: said in the output instead
            result = repairer.repair(unit, findings)
        verified = result.status == RepairStatus.VERIFIED and bool(result.repaired)
        recheck = deps.auditor.audit(dataclasses.replace(unit, javadoc=result.repaired)) if verified else None
        row = {
            "mechanism": mechanism,
            "status": result.status.value,
            "reason": result.reason,
            "verified": verified,
            "exact_rules_pass": bool(recheck) and not recheck.has_structural_debt,
            "repaired_doc": result.repaired if verified else None,
            "calls": tracker.calls - before[0],
            "cost_usd": round(tracker.cost_usd - before[1], 6),
        }
        if mechanism == "regeneration" and result.prompt_version:
            row["prompt_version"] = result.prompt_version
        out.append(row)
        reason = f" ({row['reason']})" if row["reason"] else ""
        rules = "pass" if row["exact_rules_pass"] else "FAIL"
        cost = f"{row['calls']} call(s), ${row['cost_usd']:.6f}"
        print(f"Repair     : {mechanism}: {row['status']}{reason}, exact rules {rules}, {cost}")
        if mechanism == "regeneration":
            print("             (regeneration wording: zero-shot 0, not screened for this model in the research)")
        for line in (row["repaired_doc"] or "").splitlines():
            print(f"    {line}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gpt-5.6-luna", choices=SERVICE_MODELS)
    parser.add_argument("--api-key-env", default=None, help="environment variable holding the key")
    parser.add_argument("--price-in", type=float, default=None, help="US$ per million input tokens")
    parser.add_argument("--price-out", type=float, default=None, help="US$ per million output tokens")
    parser.add_argument("--json", type=Path, default=None, help="also write the results as JSON")
    parser.add_argument(
        "--repair", action="store_true", help="also repair each method in debt: replacement, then regeneration"
    )
    args = parser.parse_args()

    provider = get_model_spec(args.model).provider
    key_env = args.api_key_env or KEY_ENV.get(provider, "OPENAI_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"Set {key_env} to the {PROVIDER_LABELS.get(provider, provider)} API key.")
    spec = build_spec(args.model, key, price_in_per_m=args.price_in, price_out_per_m=args.price_out)
    print(f"Model      : {args.model} ({PROVIDER_LABELS.get(provider, provider)}), key ...{key[-4:]}")
    screened = "screened in the research" if is_screened(args.model) else "NOT screened (zero-shot wording 0)"
    print(f"Prompt     : {screened}")

    # 1. Connection test (the Model page's "Test the connection")
    ok, error = test_connection(spec, factory=engine.CHAT_FACTORY)
    print(f"Connection : {'OK' if ok else 'REFUSED: ' + str(error)}")
    if not ok:
        raise SystemExit(2)

    # 2. One detection per method, with the application's pipeline
    tracker = UsageTracker()
    deps = build_service_deps(spec, tracker=tracker)
    units = {u.name: u for u in parse_java_source(SOURCE, "demo/Temperature.java")}
    results, failures = [], 0
    for name, (expected, case) in EXPECTED.items():
        unit = units[name]
        before = (tracker.calls, tracker.cost_usd)
        structural = deps.auditor.audit(unit).findings
        routed = deps.router.route(unit, structural)
        obtained = verdict_of(routed)
        votes = routed.final.votes if routed.final else []
        explanations = [
            {
                "field": f.field,
                "message": f.message,
                "quoted_doc": f.quoted_doc,
                "quoted_code": f.quoted_code,
                "found_word_for_word": bool(f.quoted_doc),
            }
            for f in (routed.final.semantic_findings if routed.final else [])
        ]
        row = {
            "method": name,
            "case": case,
            "expected": expected,
            "obtained": obtained,
            "match": obtained == expected,
            "rules": sorted({f.rule for f in structural if f.is_debt}),
            "model_answer": [v.raw for v in votes],
            "prompt_version": [v.prompt_version for v in votes],
            "explanations": explanations,
            "calls": tracker.calls - before[0],
            "cost_usd": round(tracker.cost_usd - before[1], 6),
        }
        results.append(row)
        failures += not row["match"]
        print(f"\n--- {name}: {case}")
        print(f"Expected   : {expected}")
        print(f"Obtained   : {obtained}  {'OK' if row['match'] else 'DIFFERENT'}")
        print(f"Rules      : {', '.join(row['rules']) or 'none'}")
        print(f"Model      : {' | '.join(str(a).strip()[:80] for a in row['model_answer']) or 'not asked'}")
        print(f"Prompt     : {', '.join(str(v) for v in row['prompt_version'] if v) or '-'}")
        for e in explanations:
            mark = "found word for word" if e["found_word_for_word"] else "not found word for word"
            print(f"Explanation: [{e['field']}] {e['message']} ({mark})")
            if e["quoted_doc"]:
                print(f"  doc  : {e['quoted_doc']}")
            if e["quoted_code"]:
                print(f"  code : {e['quoted_code']}")
        print(f"Calls/cost : {row['calls']} call(s), ${row['cost_usd']:.6f}")
        if args.repair and obtained == "debt":
            row["repairs"] = repair_both(deps, spec, tracker, unit, routed.findings)
            failures += not all(r["verified"] and r["exact_rules_pass"] for r in row["repairs"])

    tokens = f"{tracker.input_tokens} input / {tracker.output_tokens} output tokens"
    print(f"\nTotal      : {tracker.calls} call(s), {tokens}, ${tracker.cost_usd:.6f}")
    print(f"Result     : {sum(r['match'] for r in results)}/{len(results)} verdict(s) as expected")
    if args.repair:
        repairs = [x for r in results for x in r.get("repairs", [])]
        good = sum(x["verified"] and x["exact_rules_pass"] for x in repairs)
        print(f"Repairs    : {good}/{len(repairs)} verified and passing the exact rules")
    if args.json:
        args.json.write_text(json.dumps({"model": args.model, "results": results}, indent=2), encoding="utf-8")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
