"""Check that the tool reproduces the research structural results exactly.

Compares, on every benchmark item, (1) the signature extracted by the
tree-sitter parser with the research javalang extraction and (2) the
Auditor's structural-debt decision with the research ``verify_structural``.
Needs the research repository (``--research-src`` = its ``src`` folder) and
the ``dev`` extra (javalang).

Example:
    python scripts/parity_check.py --research-src /abs/path/Rewriting-the-Narrative/src \\
        --dataset /abs/path/verified_final_DEFINITIVE_363.xlsx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from docdebt.benchmark import load_items
from docdebt.core.auditor import AuditorConfig, audit_javadoc
from docdebt.core.java_parser import extract_signature_from_method_code
from docdebt.core.javadoc import simple_exception_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--research-src", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.research_src.resolve()))
    from evaluation.structural import (  # type: ignore[import-not-found]
        extract_signature_from_method_code as research_signature,
    )
    from evaluation.structural import verify_structural  # type: ignore[import-not-found]

    items = load_items(args.dataset)
    sig_ok = sig_total = audit_ok = audit_total = 0
    mismatches = []
    for item in items.to_dict("records"):
        ref = research_signature(item["method_code"])
        ours = extract_signature_from_method_code(item["method_code"])
        sig_total += 1
        if ref is None or ours is None:
            same = ref is None and ours is None
        else:
            same = (
                ref.param_names == ours.param_names
                and ref.is_void == ours.is_void
                and ref.is_constructor == ours.is_constructor
                and [simple_exception_name(t) for t in ref.throws]
                == [simple_exception_name(t) for t in ours.throws]
            )
        sig_ok += same
        if not same:
            mismatches.append((item["item_id"], "signature"))
        if ref is None or ours is None:
            continue
        audit_total += 1
        ref_debt = not verify_structural(item["stale_doc"], ref).resolved
        findings = audit_javadoc(item["stale_doc"], ours, "x", config=AuditorConfig(require_throws=True))
        our_debt = any(f.is_debt for f in findings)
        audit_ok += ref_debt == our_debt
        if ref_debt != our_debt:
            mismatches.append((item["item_id"], "audit"))

    print(f"Signatures identical: {sig_ok}/{sig_total}")
    print(f"Structural-debt decisions identical: {audit_ok}/{audit_total}")
    for item_id, what in mismatches[:20]:
        print(f"  mismatch [{what}] {item_id}")
    raise SystemExit(0 if not mismatches else 1)


if __name__ == "__main__":
    main()
