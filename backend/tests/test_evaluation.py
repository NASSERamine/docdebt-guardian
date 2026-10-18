"""The paper's metrics (docdebt.evaluation) and the ablation script, run end to end
with a scripted model."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from docdebt.detection.calibration import mcnemar_exact
from docdebt.evaluation import (
    field_texts,
    mean,
    paired_mcnemar,
    preservation,
    recall_at_k,
    reciprocal_rank,
    semantic_outcome,
    semantic_summary,
)
from docdebt.service import engine

from .conftest import KELVIN_SOURCE, make_factory
from .test_service_v3 import responder

BEFORE = "/**\n * Adds two values.\n * @param a first\n * @param b second\n * @return the sum\n */"
AFTER = "/**\n * Adds two values.\n * @param a first value\n * @param b second\n * @return the sum\n */"


def test_field_texts_split_a_javadoc_by_field():
    fields = field_texts(BEFORE)
    assert fields["@param a"] == "first" and fields["@return"] == "the sum"
    assert "description" in fields and field_texts(None) == {} and field_texts("  ") == {}


def test_preservation_counts_the_fields_the_developer_left_unchanged():
    # description, @param b and @return were left unchanged by the developer
    assert preservation(BEFORE, AFTER, AFTER) == (3, 3)
    damaged = AFTER.replace("the sum", "the total")
    assert preservation(BEFORE, AFTER, damaged) == (2, 3)
    assert preservation(BEFORE, AFTER, None) == (3, 3)  # no repair: the stale Javadoc stays


def test_semantic_outcomes_and_their_summary():
    assert semantic_outcome(0.4, 0.2) == "improved"
    assert semantic_outcome(0.4, 0.6) == "worsened"
    assert semantic_outcome(0.4, 0.4) == "unchanged" and semantic_outcome(0.4, None) == "unchanged"
    summary = semantic_summary(["improved", "improved", "worsened", "unchanged"])
    assert summary["n"] == 4 and summary["net"] == pytest.approx(0.25)
    assert semantic_summary([])["net"] is None


def test_retrieval_metrics_and_mean():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5 and reciprocal_rank(["a"], {"z"}) == 0.0
    assert recall_at_k(["a"], set(), 4) is None and reciprocal_rank(["a"], set()) is None
    assert mean([1.0, None, 0.0]) == 0.5 and mean([None]) is None


def test_paired_mcnemar_uses_only_the_items_both_configurations_ran():
    a = {"1": True, "2": True, "3": False, "4": True}
    b = {"1": True, "2": False, "3": True, "5": False}
    result = paired_mcnemar(a, b, mcnemar_exact)
    assert (result["n"], result["a_only"], result["b_only"]) == (3, 1, 1) and result["p"] == 1.0


# -- the ablation script ---------------------------------------------------------------------


STALE = "/**\n     * Converts kelvin to celsius.\n     * @param tempInCelsius temperature in kelvin\n     * @return temperature in celsius\n     */"  # noqa: E501
FIXED_DOC = STALE.replace("tempInCelsius", "tempInKelvin")
CODE = "public double toCelsius(double tempInKelvin) {\n    return tempInKelvin - 273.15;\n}"


class FlatEmbedder:
    """Stand-in for MiniLM in the summary: one dimension per known word."""

    def encode(self, texts, show_progress_bar=False):
        words = sorted({w for t in texts for w in t.lower().split()})
        return [[float(t.lower().split().count(w)) + 1e-3 for w in words] for t in texts]


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tool_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_tool_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ablation_runs_the_four_configurations_and_writes_a_summary(tmp_path, monkeypatch):
    pytest.importorskip("pandas")  # the "bench" extra
    pytest.importorskip("numpy")
    bench = load_script()
    monkeypatch.setattr(engine, "CHAT_FACTORY", make_factory(responder()))
    real = bench.cosine_distances
    monkeypatch.setattr(bench, "cosine_distances", lambda a, b: real(a, b, embedder=FlatEmbedder()))
    dataset = tmp_path / "items.csv"
    with dataset.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["item_id", "unique_id", "method_code", "stale_doc", "label", "severity",
                        "javadoc_before", "javadoc_after", "file_path"],
        )  # fmt: skip
        writer.writeheader()
        common = {"unique_id": "m1", "method_code": CODE, "javadoc_before": STALE, "javadoc_after": FIXED_DOC,
                  "file_path": "src/demo/Temperature.java"}  # fmt: skip
        writer.writerow({**common, "item_id": "m1_debt", "stale_doc": STALE, "label": "debt",
                         "severity": "critical_signature_change"})  # fmt: skip
        writer.writerow({**common, "item_id": "m1_clean", "stale_doc": FIXED_DOC, "label": "clean",
                         "severity": "none"})  # fmt: skip
    repo = tmp_path / "repos" / "m1" / "src" / "demo"
    repo.mkdir(parents=True)
    (repo / "Temperature.java").write_text(KELVIN_SOURCE, encoding="utf-8")
    (tmp_path / "repos" / "m1" / "README.md").write_text("Temperature conversions in kelvin and celsius.")
    out = tmp_path / "runs" / "tool.csv"
    monkeypatch.setenv("BENCH_KEY", "sk-test-0123456789abcd")
    argv = ["run_tool_benchmark.py", "--model", "gpt-5.6-luna", "--api-key-env", "BENCH_KEY",
            "--dataset", str(dataset), "--split", "all", "--repos-dir", str(tmp_path / "repos"),
            "--embeddings", "bm25", "--out", str(out)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    bench.main()

    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert sorted((r["item_id"], r["config"]) for r in rows) == sorted(
        (i, c) for i in ("m1_debt", "m1_clean") for c in "ABCD"
    )
    by_key = {(r["item_id"], r["config"]): r for r in rows}
    assert all(by_key[("m1_debt", c)]["verdict"] == "debt" for c in "ABCD")  # the phantom @param rule
    assert by_key[("m1_debt", "A")]["context_items"] == "0"
    assert "sibling" in by_key[("m1_debt", "B")]["context_sources"]
    assert "sibling" not in by_key[("m1_debt", "C")]["context_sources"]
    assert by_key[("m1_debt", "C")]["context_items"] != "0"
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert set(summary) == {"A", "B", "C", "D", "mcnemar_B_vs_D"}
    assert summary["D"]["detection"] and summary["D"]["items"] == 2

    bench.main()  # re-running resumes: nothing is run twice
    with out.open(newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 8
