"""Check the repair machinery on real Javadocs, with NO model.

For every structural-debt item of the benchmark, the developer's own fix
(``javadoc_after``) plays the role of the model's answer. The plan is built
from the Auditor's findings on the stale Javadoc, the fix is merged and
verified exactly as in production. What this proves:

* the merge keeps every untouched field identical (violations must be 0);
* the Verifier accepts a correct fix and reports why it rejects the others.

Items rejected here are not bugs by themselves: they show what the
developer's fix does not contain (for example an undocumented @throws), which
the Verifier correctly refuses.

Example:
    python scripts/oracle_repair_check.py --dataset datasets/verified_final_DEFINITIVE_363.xlsx
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import unified_diff
from pathlib import Path

from docdebt.benchmark import item_to_unit, load_items
from docdebt.core.auditor import Auditor
from docdebt.core.javadoc_edit import apply_edits
from docdebt.repair.plan import plan_from_findings
from docdebt.repair.verifier import Verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--show", type=int, default=2, help="Number of example diffs to print")
    args = parser.parse_args()

    items = load_items(args.dataset)
    debt = items[(items["label"] == "debt") & (items["debt_type"] == "structural")]
    auditor, verifier = Auditor(), Verifier()

    total = accepted = 0
    reasons: Counter[str] = Counter()
    preservation_violations = 0
    layouts: Counter[str] = Counter()
    shown = 0
    for item in debt.to_dict("records"):
        unit = item_to_unit(str(item["item_id"]), item["method_code"], item["stale_doc"])
        after = item.get("javadoc_after")
        if unit is None or not isinstance(after, str) or not after.strip():
            continue
        findings = auditor.audit(unit).findings
        plan = plan_from_findings(unit, findings)
        if plan.is_empty:
            continue
        total += 1
        merge = apply_edits(unit.javadoc or "", after, plan, unit.signature, base_indent=unit.indent)
        issues = verifier.verify(unit, unit.javadoc or "", plan, findings, merge)
        layouts["changed" if merge.changes else "no-change"] += 1
        preservation_violations += sum(i.code in ("preservation", "unexpected_field") for i in issues)
        if not issues:
            accepted += 1
            if shown < args.show:
                shown += 1
                print(
                    f"--- {item['item_id']}: plan add={sorted(r.label for r in plan.add)} "
                    f"remove={sorted(r.label for r in plan.remove)}"
                )
                print(
                    "\n".join(
                        unified_diff(
                            (unit.javadoc or "").splitlines(),
                            merge.text.splitlines(),
                            "original",
                            "repaired",
                            lineterm="",
                            n=1,
                        )
                    )
                )
        else:
            reasons.update({i.code for i in issues})

    print(f"\nStructural-debt items: {total}")
    print(f"Fix accepted by the Verifier: {accepted} ({accepted / total:.1%})" if total else "no items")
    print(f"Preservation violations: {preservation_violations}  (must be 0)")
    print("Rejection reasons (an item can have several):")
    for code, n in reasons.most_common():
        print(f"  {n:4d}  {code}")


if __name__ == "__main__":
    main()
