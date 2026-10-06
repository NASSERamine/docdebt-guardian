"""Measure candidate Auditor rules on the benchmark, with NO model.

A rule is only worth adding if it (a) fires on documentation debt and (b) almost
never fires on the clean Javadoc (the developers' fixed versions). For every
candidate this prints how many structural-debt, semantic-debt and clean
documents it fires on, and how many of the structural-debt items the current
Auditor misses it would catch.

Candidate rules measured:
  empty-param, empty-return    a tag with no description (already Auditor warnings)
  empty-throws                 a @throws tag with no description
  empty-undeclared-throws      same, for an exception the signature does not declare
  malformed-throws             @throws whose first token is not a class name (e.g. {@link X})

Example:
    python scripts/measure_rule_candidates.py --dataset datasets/verified_final_DEFINITIVE_363.xlsx
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from docdebt.benchmark import item_to_unit, load_items
from docdebt.core.auditor import Auditor
from docdebt.core.javadoc import parse_javadoc, simple_exception_name

THROWS_TOKEN = re.compile(r"@(?:throws|exception)\s+(\S+)")
IDENTIFIER = re.compile(r"^[A-Za-z_$][\w$]*(\.[A-Za-z_$][\w$]*)*$")


def rules_hit(unit, doc: str) -> set[str]:
    hit: set[str] = set()
    parsed = parse_javadoc(doc)
    if any(name in unit.signature.param_names and not text for name, text in parsed.params.items()):
        hit.add("empty-param")
    if parsed.return_desc == "":
        hit.add("empty-return")
    if any(not IDENTIFIER.match(token) for token in THROWS_TOKEN.findall(doc)):
        hit.add("malformed-throws")
    declared = {simple_exception_name(t) for t in unit.signature.throws}
    if any(text == "" for text in parsed.throws.values()):
        hit.add("empty-throws")
    if any(name not in declared and text == "" for name, text in parsed.throws.items()):
        hit.add("empty-undeclared-throws")
    return hit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    items = load_items(args.dataset)
    items = items[items["debt_type"].isin(["structural", "semantic", "clean"])]
    auditor = Auditor()
    totals: Counter[str] = Counter()
    fired: dict[str, Counter[str]] = {}
    missed_total, missed_by_rule = 0, Counter()
    missed_union = 0
    for item in items.to_dict("records"):
        unit = item_to_unit(str(item["item_id"]), item["method_code"], item["stale_doc"])
        if unit is None:
            continue
        kind = item["debt_type"]
        totals[kind] += 1
        hit = rules_hit(unit, item["stale_doc"])
        for rule in hit:
            fired.setdefault(rule, Counter())[kind] += 1
        if kind == "structural" and not auditor.audit(unit).has_structural_debt:
            missed_total += 1
            missed_union += bool(hit)
            missed_by_rule.update(hit)

    print(f"documents: {dict(totals)}")
    print(f"\n{'rule':<26}{'structural':>11}{'semantic':>10}{'clean':>7}{'clean rate':>12}")
    for rule, counts in sorted(fired.items()):
        rate = counts["clean"] / totals["clean"] if totals["clean"] else 0.0
        print(f"{rule:<26}{counts['structural']:>11}{counts['semantic']:>10}{counts['clean']:>7}{rate:>12.1%}")
    print(f"\nstructural-debt items the Auditor misses: {missed_total}")
    for rule, n in missed_by_rule.most_common():
        print(f"  {rule:<26} would catch {n}")
    print(f"  any of these rules          would catch {missed_union} (overlaps counted once)")


if __name__ == "__main__":
    main()
