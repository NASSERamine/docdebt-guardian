from pathlib import Path

import pytest
from typer.testing import CliRunner

from docdebt.cli import app
from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source
from docdebt.detection.calibration import (
    CalibrationRecord,
    choose_threshold,
    mcnemar_exact,
    metrics_at_threshold,
)
from docdebt.detection.citations import quote_in_code, quote_in_javadoc
from docdebt.detection.context import ContextRetriever, expand_sources
from docdebt.detection.judge import Judge, JudgeConfig, Status
from docdebt.detection.router import Router, RouterPolicy
from docdebt.llm.client import LLMClient, UsageTracker
from docdebt.llm.prompts.registry import PromptRegistry

from .conftest import KELVIN_SOURCE, make_factory

GOOD_CITATION = (
    "CLAIM: Returns the maximum of the two values.\n"
    "CODE: return Math.min(a, b);\n"
    "REASON: the code returns the minimum.\nFIELD: description\n---"
)
HALLUCINATED_CITATION = (
    "CLAIM: Sorts the list in place.\nCODE: Collections.sort(list);\nREASON: x\nFIELD: description"
)


def units():
    return {u.name: u for u in parse_java_source(KELVIN_SOURCE, "demo/Temperature.java")}


def judge_with(spec, responder, config=None, tracker=None):
    client = LLMClient(
        spec,
        factory=make_factory(responder),
        tracker=tracker or UsageTracker(),
        sleep=lambda s: None,
    )
    return Judge(client, PromptRegistry(), config or JudgeConfig())


def is_explain(prompt: str) -> bool:
    return "Find every such contradicted claim" in prompt


# -- citations ---------------------------------------------------------------


def test_quotes_are_matched_across_layout():
    doc = "/**\n * Returns the maximum of\n * the two values.\n */"
    assert quote_in_javadoc("Returns the maximum of the two values.", doc)
    assert not quote_in_javadoc("Returns the minimum", doc)
    assert not quote_in_javadoc("a", doc)  # too short to be evidence
    assert quote_in_code("return Math.min(a,  b)", "int f(){ return Math.min(a, b); }")


# -- judge -------------------------------------------------------------------


def test_judge_debt_with_validated_citation(local_spec):
    unit = units()["pick"]
    judge = judge_with(local_spec, lambda p, m: (GOOD_CITATION if is_explain(p) else "INCOMPLETE", "stop"))
    result = judge.judge(unit, [])
    assert result.status == Status.DEBT and result.reason == "cited_contradiction"
    assert result.confidence == 1.0
    finding = result.semantic_findings[0]
    assert finding.quoted_code == "return Math.min(a, b);" and finding.prompt_version.startswith("explain/")


def test_judge_rejects_hallucinated_citation(local_spec):
    unit = units()["pick"]
    judge = judge_with(local_spec, lambda p, m: (HALLUCINATED_CITATION if is_explain(p) else "INCOMPLETE", "stop"))
    result = judge.judge(unit, [])
    assert result.status == Status.ABSTAIN and result.reason == "unsupported_verdict"
    assert result.rejected_citations == 1 and not result.semantic_findings


def test_judge_structural_support(local_spec):
    unit = units()["toCelsius"]
    structural = Auditor().audit(unit).findings
    judge = judge_with(local_spec, lambda p, m: ("NONE" if is_explain(p) else "INCOMPLETE", "stop"))
    result = judge.judge(unit, structural)
    assert result.status == Status.DEBT and result.reason == "structural_support"


def test_judge_clean_unparsable_and_tie(local_spec):
    unit = units()["pick"]
    assert judge_with(local_spec, lambda p, m: ("COMPLETE", "stop")).judge(unit).status == Status.CLEAN
    abstain = judge_with(local_spec, lambda p, m: ("no idea", "stop")).judge(unit)
    assert abstain.status == Status.ABSTAIN and abstain.reason == "no_valid_vote"

    def split(prompt, m):  # zeroshot says debt, cot says clean
        return ("Final Answer: COMPLETE" if "step by step" in prompt else "INCOMPLETE", "stop")

    tie = judge_with(local_spec, split, JudgeConfig(patterns=["zeroshot", "cot"])).judge(unit)
    assert tie.status == Status.ABSTAIN and tie.reason == "tie"


def test_judge_confidence_threshold(local_spec):
    answers = iter(["INCOMPLETE", "INCOMPLETE", "COMPLETE"])
    unit = units()["pick"]

    def responder(prompt, m):
        return (GOOD_CITATION if is_explain(prompt) else next(answers), "stop")

    config = JudgeConfig(samples_per_pattern=3, confidence_threshold=0.8)
    result = judge_with(local_spec, responder, config).judge(unit)
    assert result.confidence == pytest.approx(2 / 3)
    assert result.status == Status.ABSTAIN and result.reason == "low_confidence"


# -- router ------------------------------------------------------------------


def test_router_escalates_only_when_allowed(local_spec, cloud_spec, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    unit = units()["pick"]
    tracker = UsageTracker()
    local = judge_with(local_spec, lambda p, m: ("unsure", "stop"), tracker=tracker)
    cloud = judge_with(
        cloud_spec,
        lambda p, m: (GOOD_CITATION if is_explain(p) else "INCOMPLETE", "stop"),
        tracker=tracker,
    )
    judges = {"qwen2.5-coder:7b": local, "gpt-5.6": cloud}

    private = Router(judges, RouterPolicy(cascade=list(judges), allow_cloud=False), tracker)
    result = private.route(unit, [])
    assert result.status == Status.ABSTAIN and result.skipped_models == ["gpt-5.6"]

    hybrid = Router(judges, RouterPolicy(cascade=list(judges), allow_cloud=True), tracker)
    result = hybrid.route(unit, [])
    assert result.status == Status.DEBT and result.escalated
    assert [t.model_id for t in result.trail] == ["qwen2.5-coder:7b", "gpt-5.6"]

    capped = Router(judges, RouterPolicy(cascade=list(judges), allow_cloud=True, max_tokens_total=1), tracker)
    assert capped.route(unit, []).note == "budget_exhausted"


def test_router_skip_llm_when_structural(local_spec):
    unit = units()["toCelsius"]
    judge = judge_with(local_spec, lambda p, m: pytest.fail("LLM must not be called"))
    router = Router(
        {"qwen2.5-coder:7b": judge},
        RouterPolicy(cascade=["qwen2.5-coder:7b"], skip_llm_when_structural=True),
    )
    result = router.route(unit, Auditor().audit(unit).findings)
    assert result.status == Status.DEBT and result.note == "structural_only"


# -- context -----------------------------------------------------------------


def test_context_callees_exclude_target_and_tests(tmp_path: Path):
    main = tmp_path / "src" / "main" / "java" / "Calc.java"
    test = tmp_path / "src" / "test" / "java" / "CalcTest.java"
    main.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    main.write_text(
        "class Calc {\n"
        "  /** Sum of a list. @param xs values @return sum */\n"
        "  int sum(int[] xs) { return sum(xs, 0); }\n"
        "  /** CORRECTED DOC OF THE OVERLOAD @param xs v @param i start @return s */\n"
        "  int sum(int[] xs, int i) { return i >= xs.length ? 0 : xs[i] + sum(xs, i + 1) + clamp(i); }\n"
        "  /** Clamp. @param v value @return clamped */\n"
        "  int clamp(int v) { return v; }\n"
        "}\n",
        encoding="utf-8",
    )
    test.write_text(
        "class CalcTest { void t() { new Calc().sum(new int[]{1}); } void u() {} }",
        encoding="utf-8",
    )
    target = next(
        u
        for u in parse_java_source(main.read_text(), "src/main/java/Calc.java")
        if u.signature.param_names == ["xs", "i"]
    )
    ctx = ContextRetriever(tmp_path, ["siblings_called", "tests"]).retrieve(target)
    assert "Clamp." in ctx  # a real callee is kept
    assert "CORRECTED DOC OF THE OVERLOAD" not in ctx  # the leak is closed
    assert "--- Test exercising this method ---" in ctx and "void u()" not in ctx
    assert expand_sources(["siblings_tests", "siblings"]) == ("siblings", "tests")


# -- calibration -------------------------------------------------------------


def test_calibration_and_mcnemar():
    records = [
        CalibrationRecord("1", True, "debt", 1.0),
        CalibrationRecord("2", True, "debt", 0.6),
        CalibrationRecord("3", False, "debt", 0.6),
        CalibrationRecord("4", False, "clean", 1.0),
    ]
    at0 = metrics_at_threshold(records, 0.0)
    assert at0.precision == pytest.approx(2 / 3) and at0.coverage == 1.0
    chosen = choose_threshold(records, target_precision=0.9, min_alerts=1)
    assert chosen.threshold == 1.0 and chosen.precision == 1.0 and chosen.coverage == 0.5
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


# -- CLI ---------------------------------------------------------------------


def test_cli_check(tmp_path: Path):
    java = tmp_path / "Temperature.java"
    java.write_text(KELVIN_SOURCE, encoding="utf-8")
    result = CliRunner().invoke(app, ["check", str(java)])
    assert result.exit_code == 1
    assert "[unknown-param]" in result.output and "3 method(s) checked, 2 with structural debt" in result.output
    ok = CliRunner().invoke(app, ["check", str(java), "--fail-on", "never", "--format", "json"])
    assert ok.exit_code == 0 and '"rule": "missing-throws"' in ok.output


def test_pipeline_end_to_end(tmp_path: Path):
    from docdebt.pipeline import build_detection_pipeline
    from docdebt.settings import load_settings

    (tmp_path / ".docdebt.yml").write_text(
        "router:\n  cascade: [qwen2.5-coder:7b]\njudges:\n  qwen2.5-coder:7b:\n    patterns: [zeroshot]\n",
        encoding="utf-8",
    )
    settings = load_settings(search_from=tmp_path)

    def responder(prompt, m):
        if is_explain(prompt):
            return (GOOD_CITATION if "Math.min" in prompt else "NONE", "stop")
        return (
            "INCOMPLETE" if ("Math.min" in prompt or "tempInCelsius" in prompt) else "COMPLETE",
            "stop",
        )

    pipeline = build_detection_pipeline(settings, factory=make_factory(responder))
    results = {r.unit.name: r for r in pipeline.run(list(units().values())[:3])}
    assert results["toCelsius"].status == Status.DEBT  # structural support
    assert results["pick"].status == Status.DEBT  # cited semantic contradiction
    # The LLM says clean, but the undocumented IOException is an exact structural error.
    assert results["read"].semantic_status == Status.CLEAN
    assert results["read"].status == Status.DEBT
    assert pipeline.tracker.calls > 0


def test_quote_in_code_handles_concatenated_multiline_strings():
    from docdebt.detection.citations import quote_in_code

    real_source = """throw new DeprecationException(
                "Iterators on files are not supported any more. " +
                        "Use File.eachLine() instead. Alternatively you can use FileReader.iterator() " +
                        "and provide your own exception handling."
        );"""
    model_quote = (
        'throw new DeprecationException("Iterators on files are not supported any more. '
        "Use File.eachLine() instead. Alternatively you can use FileReader.iterator() "
        'and provide your own exception handling.");'
    )
    assert quote_in_code(model_quote, real_source)
    assert not quote_in_code("return Math.max(a, b);", real_source)
