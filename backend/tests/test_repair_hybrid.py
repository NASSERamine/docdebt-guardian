"""Hybrid Replacement repair (research layers 1-4) behind the Repairer interface."""

import json

from docdebt.core.java_parser import parse_java_source
from docdebt.core.models import Finding, FindingKind, Severity
from docdebt.llm.client import LLMClient
from docdebt.llm.config import ModelSpec
from docdebt.repair import hybrid
from docdebt.repair.brace_tokens import decode_braces, encode_braces, parse_batched_json_patch
from docdebt.repair.hybrid_repairer import HybridRepairer, prompt_version
from docdebt.repair.repairer import RepairStatus
from docdebt.repair.spans import apply_span_patch, parse_javadoc_spans

from .conftest import KELVIN_SOURCE, make_factory

SPEC = ModelSpec(id="qwen2.5-coder:7b", provider="ollama", model="qwen2.5-coder:7b", max_tokens={"repair": 1400})


def units():
    return {u.name: u for u in parse_java_source(KELVIN_SOURCE, "src/Temperature.java")}


def structural(unit, rule="unknown-param"):
    return [Finding(FindingKind.STRUCTURAL, Severity.ERROR, rule, "debt", unit.id)]


def repairer(answer, calls=None):
    """A repairer whose model answers ``answer`` (a dict, sent as JSON, or raw text)."""

    def respond(prompt, _max_tokens):
        if calls is not None:
            calls.append(prompt)
        text = answer(prompt) if callable(answer) else answer
        return (json.dumps(text) if isinstance(text, dict) else text), "stop"

    client = LLMClient(SPEC, factory=make_factory(respond), sleep=lambda s: None)
    return HybridRepairer(client)


def test_spans_patch_keeps_untouched_lines_byte_identical():
    doc = "/**\n     * Says hi.\n     * <p>Long {@link Foo}.\n     * @param a the a\n     * @see Bar\n     */"
    parsed = parse_javadoc_spans(doc)
    assert [f.key for f in parsed.fields] == ["description", "@param a", "@see"]
    out = apply_span_patch(parsed, replace={"@param a": "the first value"})
    assert out.splitlines()[:3] == doc.splitlines()[:3] and "@see Bar" in out
    assert "@param a the first value" in out


def test_brace_tokens_roundtrip_and_tolerant_json_parsing():
    text = "uses {@link Foo#bar()} and {@code x}"
    assert decode_braces(encode_braces(text)) == text and "{" not in encode_braces(text)
    assert parse_batched_json_patch('```json\n{"m": {"@return": "keep"}}\n```') == {"m": {"@return": "keep"}}
    assert parse_batched_json_patch("no json here") is None


def test_rename_is_deterministic_and_the_model_only_judges():
    unit = units()["toCelsius"]
    calls = []
    result = repairer({"m": {"description": "keep", "@return": "keep"}}, calls).repair(unit, structural(unit))
    assert result.status == RepairStatus.VERIFIED and result.mechanism == "hybrid"
    assert "@param tempInKelvin temperature in kelvin" in result.repaired  # text moved verbatim
    assert "tempInCelsius" not in result.repaired
    assert [c.ref.label for c in result.changes] == ["@param tempInCelsius", "@param tempInKelvin"]
    # every other line is byte-identical to the original
    kept = [line for line in unit.javadoc.splitlines() if "tempInCelsius" not in line]
    assert kept == [line for line in result.repaired.splitlines() if "tempInKelvin" not in line]
    assert len(calls) == 1 and "FIELDS TO JUDGE" in calls[0] and "tempInKelvin" in calls[0]
    assert result.prompt_version == prompt_version()


def test_a_cited_semantic_debt_is_judged_and_corrected():
    unit = units()["pick"]
    finding = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "semantic",
        "The return value is described wrongly.",
        unit.id,
        field="@return",
        quoted_doc="the smaller of a and b",
        quoted_code="Math.min(a, b)",
    )
    calls = []
    result = repairer({"m": {"description": "Returns the minimum of the two values.", "@return": "keep"}}, calls)
    result = result.repair(unit, [finding])
    assert result.status == RepairStatus.VERIFIED
    assert "Returns the minimum of the two values." in result.repaired
    assert "The documentation says" in calls[0]  # the citation is a hint for the model
    assert {e["key"]: e["outcome"] for e in result.field_log}["description"] == "accepted"


def test_post_validation_reverts_invented_identifiers_and_cosmetic_edits():
    unit = units()["pick"]
    findings = [
        Finding(FindingKind.SEMANTIC, Severity.ERROR, "semantic", "wrong", unit.id, field="description"),
        Finding(FindingKind.SEMANTIC, Severity.ERROR, "semantic", "wrong", unit.id, field="@return"),
    ]
    answer = {"m": {"description": "Returns the max using `someHelperCall`.", "@return": "The smaller of a and b"}}
    result = repairer(answer).repair(unit, findings)
    outcomes = {e["key"]: e["outcome"] for e in result.field_log}
    assert outcomes["description"] == "reverted_hallucination"
    assert outcomes["@return"] == "reverted_cosmetic"
    assert result.status == RepairStatus.UNCHANGED  # nothing left to propose


def test_a_missing_param_is_written_and_a_failed_write_is_retried_once():
    source = """class A {
    /**
     * Adds.
     * @param a first
     * @return the sum
     */
    public int add(int a, int b) { return a + b; }
}"""
    unit = parse_java_source(source, "A.java")[0]
    answers = iter([{"m": {"description": "keep"}}, {"m": {"@param b": "the second operand"}}])
    calls = []
    result = repairer(lambda _p: next(answers), calls).repair(unit, structural(unit, "missing-param"))
    assert len(calls) == 2 and "FIELDS TO WRITE (" in calls[1] and "FIELDS TO JUDGE (" not in calls[1]
    assert result.status == RepairStatus.VERIFIED and "@param b the second operand" in result.repaired
    # the new tag is placed after @param a, before @return
    lines = [ln.strip() for ln in result.repaired.splitlines()]
    assert lines.index("* @param b the second operand") == lines.index("* @param a first") + 1


def test_a_write_the_model_never_answers_fails_the_structural_check():
    source = "class A {\n    /**\n     * Adds.\n     */\n    public int add(int a) { return a; }\n}"
    unit = parse_java_source(source, "A.java")[0]
    result = repairer({"m": {}}).repair(unit, structural(unit, "missing-param"))
    assert result.status == RepairStatus.FAILED and result.reason.startswith("structure:")
    assert result.repaired is None


def test_an_answer_without_the_item_level_is_still_read():
    unit = units()["toCelsius"]
    result = repairer({"description": "keep", "@return": "keep"}).repair(unit, structural(unit))
    assert result.status == RepairStatus.VERIFIED


def test_context_is_given_to_the_model_as_reference_only():
    unit = units()["toCelsius"]
    calls = []
    repairer({"m": {"description": "keep", "@return": "keep"}}, calls).repair(
        unit, structural(unit), context="--- Similar method: X.y ---\n/** Converts {@link T}. */"
    )
    assert "PROJECT CONTEXT" in calls[0] and "never return it" in calls[0]
    assert "{@link" not in calls[0].split("PROJECT CONTEXT")[1]  # braces encoded in the context too


def test_no_debt_finding_means_no_model_call():
    unit = units()["toCelsius"]
    calls = []
    result = repairer({"m": {}}, calls).repair(unit, [])
    assert result.status == RepairStatus.NOT_NEEDED and calls == []


def test_llm_error_is_reported_as_failed():
    unit = units()["toCelsius"]

    def broken(_prompt, _max):
        return RuntimeError("server down")

    client = LLMClient(SPEC, factory=make_factory(broken), sleep=lambda s: None)
    result = HybridRepairer(client).repair(unit, structural(unit))
    assert result.status == RepairStatus.FAILED and result.reason == "llm_error"


def test_judge_variants_flagged_and_all():
    unit = units()["read"]
    flagged = hybrid.prepare_hybrid_item("m", unit.code, unit.javadoc, judge="flagged")
    every = hybrid.prepare_hybrid_item("m", unit.code, unit.javadoc, judge="all")
    assert {j["key"] for j in flagged["judge"]} <= {j["key"] for j in every["judge"]}
    assert "description" in {j["key"] for j in flagged["judge"]}
    assert {w["key"] for w in every["write"]} == {"@throws IOException"}  # declared, undocumented
