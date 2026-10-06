import json
import warnings
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docdebt.cli import app
from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source, parse_method_snippet
from docdebt.core.javadoc_edit import (
    DESCRIPTION,
    PARAM,
    RETURN,
    THROWS,
    EditPlan,
    FieldRef,
    apply_edits,
    parse_layout,
    render_layout,
)
from docdebt.core.models import Finding, FindingKind, Severity
from docdebt.llm.client import LLMClient
from docdebt.repair.apply import apply_repairs_to_file, javadoc_diff
from docdebt.repair.extract import extract_javadoc_block
from docdebt.repair.fewshot import RepairFewShotSet
from docdebt.repair.plan import field_ref_from_text, plan_from_findings
from docdebt.repair.registry import RepairRegistry
from docdebt.repair.repairer import RepairConfig, Repairer, RepairStatus
from docdebt.repair.verifier import Verifier, check_preservation
from docdebt.repair.wordings import RegenExample

from .conftest import KELVIN_SOURCE, make_factory

SIG = parse_method_snippet("public int f(int a, int b) throws IOException { return a; }").signature

DOC = """/**
     * Adds two numbers.
     *
     * @param a first value
     * @param b second value
     * @return the sum
     * @see Other
     */"""


def units():
    return {u.name: u for u in parse_java_source(KELVIN_SOURCE, "demo/Temperature.java")}


def model_block(*lines: str) -> str:
    return "/**\n" + "\n".join(f" * {line}" for line in lines) + "\n */"


# -- javadoc_edit ------------------------------------------------------------


def test_layout_roundtrip_is_byte_identical():
    for doc in (DOC, "/**\n\t * Tabs.\n\t * @param a x\n\t */", "/**\r\n * Crlf.\r\n * @return r\r\n */\r\n"):
        assert render_layout(parse_layout(doc)) == doc


def test_rename_reuses_the_slot_and_ignores_model_rewrites():
    original = """/**
     * Converts.
     * @param oldName temperature
     * @return celsius
     * @see Other
     */"""
    plan = EditPlan(add={FieldRef(PARAM, "a")}, remove={FieldRef(PARAM, "oldName")})
    merged = apply_edits(original, model_block("REWRITTEN", "@param a new text", "@return REWRITTEN"), plan, SIG)
    assert merged.text == original.replace("@param oldName temperature", "@param a new text")
    assert [c.action for c in merged.changes] == ["remove", "add"]


def test_new_tags_are_ordered_by_signature():
    original = "/**\n * Adds.\n *\n * @param b second\n * @see Other\n */"
    plan = EditPlan(add={FieldRef(PARAM, "a"), FieldRef(RETURN), FieldRef(THROWS, "IOException")})
    model = model_block("x", "@param a first", "@return sum", "@throws java.io.IOException if it fails")
    text = apply_edits(original, model, plan, SIG).text
    tags = [line.strip() for line in text.splitlines() if "@" in line]
    assert tags == [
        "* @param a first",
        "* @param b second",
        "* @return sum",
        "* @throws java.io.IOException if it fails",
        "* @see Other",
    ]


def test_first_tag_on_description_only_doc_gets_a_blank_separator():
    original = "/**\n * Just a description.\n */"
    plan = EditPlan(add={FieldRef(PARAM, "a")})
    text = apply_edits(original, model_block("d", "@param a value"), plan, SIG).text
    assert text == "/**\n * Just a description.\n *\n * @param a value\n */"


def test_single_line_comment_is_rendered_in_standard_layout():
    plan = EditPlan(add={FieldRef(RETURN)})
    merged = apply_edits(
        "/** Adds. @param a x @param b y */", model_block("d", "@return sum"), plan, SIG, base_indent="    "
    )
    assert merged.text == "/**\n     * Adds.\n     * @param a x\n     * @param b y\n     * @return sum\n     */"


def test_replace_keeps_trailing_blank_and_other_blocks_verbatim():
    plan = EditPlan(replace={FieldRef(DESCRIPTION)})
    merged = apply_edits(DOC, model_block("Adds two integers together."), plan, SIG)
    assert "Adds two integers together." in merged.text and "Adds two numbers." not in merged.text
    old, new = DOC.splitlines(), merged.text.splitlines()
    assert old[2:] == new[2:]  # blank separator and every tag line are byte-identical


def test_missing_and_unchanged_fields_are_reported():
    plan = EditPlan(add={FieldRef(THROWS, "IOException")}, replace={FieldRef(PARAM, "a")})
    merged = apply_edits(DOC, model_block("x", "@param a first value"), plan, SIG)
    assert merged.missing_in_model == [FieldRef(THROWS, "IOException")]
    assert merged.unchanged == [FieldRef(PARAM, "a")] and merged.text == DOC


def test_plan_normalize_resolves_overlaps():
    ref = FieldRef(PARAM, "a")
    plan = EditPlan(add={ref}, replace={ref}, remove={FieldRef(PARAM, "z")}).normalize()
    assert plan.replace == set() and plan.add == {ref}


# -- extraction, plan, registry ---------------------------------------------


def test_extract_block_variants():
    block = "/**\n * Doc.\n */"
    assert extract_javadoc_block(f"Sure!\n```java\n{block}\n```\nDone") == block
    assert extract_javadoc_block(f"/** reasoning example */ ... Final Answer:\n{block}") == block
    assert extract_javadoc_block("no comment here") is None and extract_javadoc_block(None) is None


def test_plan_from_structural_and_semantic_findings():
    unit = units()["toCelsius"]
    findings = Auditor().audit(unit).findings
    plan = plan_from_findings(unit, findings)
    assert plan.add == {FieldRef(PARAM, "tempInKelvin")} and plan.remove == {FieldRef(PARAM, "tempInCelsius")}
    pick = units()["pick"]
    assert field_ref_from_text("@param a", pick) == FieldRef(PARAM, "a")
    assert field_ref_from_text("@param ghost", pick) == FieldRef(DESCRIPTION)
    assert field_ref_from_text("@throws java.io.IOException", pick) == FieldRef(THROWS, "IOException")
    assert (
        field_ref_from_text("@return", pick) == FieldRef(RETURN)
        and field_ref_from_text(None, pick).kind == DESCRIPTION
    )


def test_registry_locked_unscreened_and_feedback():
    registry = RepairRegistry()
    assert registry.resolve("qwen2.5-coder:7b", "zeroshot") == (2, True)
    assert registry.resolve("deepseek-coder:6.7b-instruct", "cot") == (1, True)
    with pytest.raises(KeyError, match="--wording"):
        registry.resolve("gpt-5.6", "zeroshot")
    with pytest.raises(ValueError):
        registry.resolve("qwen2.5-coder:7b", "persona")
    with pytest.raises(ValueError, match="few-shot"):
        registry.build("qwen2.5-coder:7b", "oneshot", "c", "d")

    prompt = registry.build("qwen2.5-coder:7b", "zeroshot", "CODE", "DOC", feedback=["Document 'a'."])
    assert prompt.screened and prompt.version.startswith("repair/zeroshot/w2/")
    assert prompt.text.endswith("Answer again in the same format and fix these problems.")
    override = RepairRegistry(overrides={"gpt-5.6": {"zeroshot": 0}})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert not override.build("gpt-5.6", "zeroshot", "c", "d").screened
    assert any("not screened" in str(w.message) for w in caught)


def test_repair_fewshot_uses_explicit_positions(tmp_path: Path):
    payload = {
        "examples": [{"method_code": f"c{i}", "stale_doc": f"s{i}", "repaired_doc": f"r{i}"} for i in range(3)],
        "oneshot": [1],
        "threeshot": [0, 1, 2],
    }
    path = tmp_path / "fs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    fewshot = RepairFewShotSet.load(path)
    assert [e.method_code for e in fewshot.for_pattern("oneshot")] == ["c1"]
    assert "c1" in RepairRegistry(fewshot=fewshot).build("qwen2.5-coder:7b", "oneshot", "x", "y").text
    with pytest.raises(ValueError):
        RepairFewShotSet([RegenExample("a", "b", "c")], [3], [0])


# -- verifier ----------------------------------------------------------------


def test_verifier_flags_tampering_empty_fields_and_lingering_claims():
    unit = units()["toCelsius"]
    findings = Auditor().audit(unit).findings
    plan = plan_from_findings(unit, findings)
    good = apply_edits(unit.javadoc, model_block("x", "@param tempInKelvin kelvin"), plan, unit.signature)
    assert Verifier().verify(unit, unit.javadoc, plan, findings, good) == []

    tampered = good.text.replace("Converts kelvin to celsius.", "Changed!")
    assert {i.code for i in check_preservation(unit.javadoc, tampered, plan)} == {"preservation"}

    empty = apply_edits(unit.javadoc, model_block("x", "@param tempInKelvin"), plan, unit.signature)
    assert "empty_field" in {i.code for i in Verifier().verify(unit, unit.javadoc, plan, findings, empty)}

    missing = apply_edits(unit.javadoc, model_block("x", "@return y"), plan, unit.signature)
    assert "missing_in_output" in {i.code for i in Verifier().verify(unit, unit.javadoc, plan, findings, missing)}

    pick = units()["pick"]
    finding = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "contradicted-claim",
        "m",
        pick.id,
        field="description",
        quoted_doc="Returns the maximum of the two values.",
        quoted_code="return Math.min(a, b);",
    )
    plan = plan_from_findings(pick, [finding])
    same_claim = apply_edits(
        pick.javadoc, model_block("Returns the  maximum of the two values, sorted."), plan, pick.signature
    )
    assert same_claim.changes and "claim_still_present" not in {
        i.code for i in Verifier().verify(pick, pick.javadoc, plan, [finding], same_claim)
    }  # the claim wording changed, so the exact quote is gone


# -- repairer ----------------------------------------------------------------


def repairer_with(spec, responder, config=None, calls=None):
    client = LLMClient(spec, factory=make_factory(responder, calls), sleep=lambda s: None)
    return Repairer(client, RepairRegistry(), config=config or RepairConfig())


def test_repairer_structural_happy_path_ignores_model_rewrites(local_spec):
    unit = units()["toCelsius"]
    answer = (
        "```java\n"
        + model_block(
            "MODEL REWROTE THIS", "@param tempInKelvin temperature in kelvin", "@return MODEL REWROTE THIS TOO"
        )
        + "\n```"
    )
    prompts: list[str] = []

    def responder(prompt, m):
        prompts.append(prompt)
        return answer, "stop"

    repairer = repairer_with(local_spec, responder)
    result = repairer.repair(unit, Auditor().audit(unit).findings)
    assert result.status == RepairStatus.VERIFIED and len(result.attempts) == 1
    assert "MODEL REWROTE" not in result.repaired and "Converts kelvin to celsius." in result.repaired
    assert "@return temperature in celsius" in result.repaired
    assert "CURRENT DOCUMENTATION:" in prompts[0] and "rejected" not in prompts[0]


def test_repairer_retries_with_feedback_then_succeeds(local_spec):
    unit = units()["read"]
    findings = Auditor().audit(unit).findings
    answers = iter(
        [model_block("x", "@return only"), model_block("x", "@throws java.io.IOException if the read fails")]
    )
    prompts: list[str] = []

    def responder(prompt, m):
        prompts.append(prompt)
        return next(answers), "stop"

    result = repairer_with(local_spec, responder).repair(unit, findings)
    assert result.status == RepairStatus.VERIFIED and len(result.attempts) == 2
    assert "rejected for these reasons" in prompts[1] and "@throws tag" in prompts[1]
    assert "@throws java.io.IOException if the read fails" in result.repaired


def test_repairer_fails_after_max_attempts_and_never_proposes(local_spec):
    unit = units()["read"]
    result = repairer_with(local_spec, lambda p, m: (model_block("nothing useful"), "stop")).repair(
        unit, Auditor().audit(unit).findings
    )
    assert result.status == RepairStatus.FAILED and result.repaired is None
    assert result.reason == "missing_in_output" and len(result.attempts) == 2


def test_repairer_truncated_no_block_and_llm_errors(local_spec):
    unit = units()["read"]
    findings = Auditor().audit(unit).findings
    cut = repairer_with(local_spec.model_copy(update={"max_tokens_cap": 1400}), lambda p, m: ("/** cut", "length"))
    assert cut.repair(unit, findings).reason == "truncated"
    assert (
        repairer_with(local_spec, lambda p, m: ("no block", "stop")).repair(unit, findings).reason == "no_javadoc"
    )
    down = repairer_with(local_spec, lambda p, m: ConnectionError("down"))
    result = down.repair(unit, findings)
    assert result.status == RepairStatus.FAILED and result.llm_failed and result.reason == "llm_error"


def test_repairer_semantic_repair_and_unchanged(local_spec):
    pick = units()["pick"]
    finding = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "contradicted-claim",
        "min, not max",
        pick.id,
        field="description",
        quoted_doc="Returns the maximum of the two values.",
        quoted_code="return Math.min(a, b);",
    )
    fixed = model_block("Returns the minimum of the two values.")
    result = repairer_with(local_spec, lambda p, m: (fixed, "stop")).repair(pick, [finding])
    assert result.status == RepairStatus.VERIFIED and "minimum" in result.repaired
    assert "@return the smaller of a and b" in result.repaired

    kept = model_block("Returns the maximum of the two values.")
    result = repairer_with(local_spec, lambda p, m: (kept, "stop")).repair(pick, [finding])
    assert result.status == RepairStatus.UNCHANGED and result.repaired is None


def test_repairer_skips_when_nothing_to_do(local_spec):
    pick = units()["pick"]
    repairer = repairer_with(local_spec, lambda p, m: pytest.fail("the model must not be called"))
    assert repairer.repair(pick, []).status == RepairStatus.NOT_NEEDED
    no_doc = units()["helper"]
    finding = Finding(FindingKind.STRUCTURAL, Severity.ERROR, "missing-param", "m", no_doc.id, field="@param p")
    assert repairer.repair(no_doc, [finding]).status == RepairStatus.SKIPPED


def test_hint_findings_only_in_first_prompt(local_spec):
    unit = units()["read"]
    prompts: list[str] = []

    def responder(prompt, m):
        prompts.append(prompt)
        return model_block("x"), "stop"

    repairer_with(local_spec, responder, RepairConfig(hint_findings=True)).repair(
        unit, Auditor().audit(unit).findings
    )
    assert "found these problems" in prompts[0] and "found these problems" not in prompts[1]


# -- writing and CLI ---------------------------------------------------------


def test_apply_to_file_writes_and_refuses_stale_files(tmp_path: Path, local_spec):
    java = tmp_path / "Temperature.java"
    java.write_text(KELVIN_SOURCE, encoding="utf-8")
    unit = next(u for u in parse_java_source(KELVIN_SOURCE, "Temperature.java") if u.name == "toCelsius")
    answer = model_block("x", "@param tempInKelvin temperature in kelvin")
    result = repairer_with(local_spec, lambda p, m: (answer, "stop")).repair(unit, Auditor().audit(unit).findings)
    assert "@@" in javadoc_diff(unit, result)

    written, skipped = apply_repairs_to_file(java, [(unit, result)])
    assert (written, skipped) == (1, [])
    new_source = java.read_text(encoding="utf-8")
    assert "@param tempInKelvin temperature in kelvin" in new_source and "tempInCelsius" not in new_source
    assert new_source.count("\n") == KELVIN_SOURCE.count("\n")  # same number of lines, nothing else moved
    assert [u.name for u in parse_java_source(new_source)] == ["toCelsius", "pick", "read", "helper"]
    assert not Auditor().audit(parse_java_source(new_source)[0]).has_structural_debt

    written, skipped = apply_repairs_to_file(java, [(unit, result)])  # the file changed since the analysis
    assert written == 0 and "file changed" in skipped[0]


def test_cli_repair_end_to_end(tmp_path: Path, monkeypatch):
    java = tmp_path / "Temperature.java"
    java.write_text(KELVIN_SOURCE, encoding="utf-8")

    def responder(prompt, m):
        if "toCelsius" in prompt:
            return model_block("x", "@param tempInKelvin temperature in kelvin"), "stop"
        return model_block("x", "@throws java.io.IOException if the file cannot be read"), "stop"

    monkeypatch.setattr("docdebt.llm.client.default_chat_model_factory", make_factory(responder))
    runner = CliRunner()
    shown = runner.invoke(app, ["repair", str(java)])
    assert shown.exit_code == 0, shown.output
    assert "toCelsius -> verified" in shown.output and "read -> verified" in shown.output
    assert "+     * @param tempInKelvin temperature in kelvin" in shown.output
    assert java.read_text(encoding="utf-8") == KELVIN_SOURCE  # nothing written without --write

    as_json = json.loads(runner.invoke(app, ["repair", str(java), "--format", "json"]).output)
    assert {r["status"] for r in as_json} == {"verified"} and as_json[0]["changes"]

    done = runner.invoke(app, ["repair", str(java), "--write"])
    assert "2 Javadoc(s) written." in done.output
    assert not Auditor().audit_all(parse_java_source(java.read_text(encoding="utf-8")))[0].has_structural_debt
    again = runner.invoke(app, ["repair", str(java)])
    assert "0 with debt" in again.output


def test_cli_reports_a_missing_path(tmp_path: Path):
    result = CliRunner().invoke(app, ["repair", str(tmp_path / "missing.java")])
    assert result.exit_code == 2 and "Path not found" in result.output
    assert CliRunner().invoke(app, ["check", str(tmp_path / "missing.java")]).exit_code == 2


def test_build_repairer_hint_override():
    from docdebt.pipeline import build_repairer
    from docdebt.settings import Settings

    assert build_repairer(Settings()).config.hint_findings is False
    assert build_repairer(Settings(), hint_findings=True).config.hint_findings is True


def test_cli_semantic_repair_with_hints_and_merged_usage(tmp_path: Path, monkeypatch):
    java = tmp_path / "Only.java"
    java.write_text(
        "class Only {\n    /**\n     * Returns the maximum of the two values.\n     * @param a first\n"
        "     * @param b second\n     * @return the smaller of a and b\n     */\n"
        "    public int pick(int a, int b) { return Math.min(a, b); }\n}\n",
        encoding="utf-8",
    )
    prompts: list[str] = []

    def responder(prompt, m):
        if "Find every such contradicted claim" in prompt:
            return (
                "CLAIM: Returns the maximum of the two values.\nCODE: return Math.min(a, b);\n"
                "REASON: it returns the minimum\nFIELD: description\n---",
                "stop",
            )
        if "CURRENT DOCUMENTATION:" in prompt:
            prompts.append(prompt)
            return model_block("Returns the minimum of the two values."), "stop"
        return "INCOMPLETE", "stop"

    monkeypatch.setattr("docdebt.llm.client.default_chat_model_factory", make_factory(responder))
    result = CliRunner().invoke(app, ["repair", str(java), "--semantic", "--hint-findings"])
    assert result.exit_code == 0, result.output
    assert "Only.pick -> verified" in result.output and "+     * Returns the minimum" in result.output
    assert "found these problems" in prompts[0] and "Returns the maximum of the two values." in prompts[0]
    assert "3 LLM call(s), 36 tokens" in result.output  # vote + citation + repair, all counted


def test_semantic_field_is_located_from_the_quote_not_from_the_models_label(local_spec):
    pick = units()["pick"]
    finding = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "contradicted-claim",
        "min, not max",
        pick.id,
        field="@return",  # the model labelled the wrong field
        quoted_doc="Returns the maximum of the two values.",
        quoted_code="return Math.min(a, b);",
    )
    assert plan_from_findings(pick, [finding]).replace == {FieldRef(DESCRIPTION)}
    fixed = model_block("Returns the minimum of the two values.", "@return the smaller of a and b")
    result = repairer_with(local_spec, lambda p, m: (fixed, "stop")).repair(pick, [finding])
    assert result.status == RepairStatus.VERIFIED and "minimum" in result.repaired
    assert "@return the smaller of a and b" in result.repaired  # the correct @return is untouched

    # the model's label is kept when it is right, and when no field contains the quote
    right = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "c",
        "m",
        pick.id,
        field="description",
        quoted_doc="Returns the maximum of the two values.",
    )
    assert plan_from_findings(pick, [right]).replace == {FieldRef(DESCRIPTION)}
    unknown = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "c",
        "m",
        pick.id,
        field="@return",
        quoted_doc="text that is nowhere in the javadoc",
    )
    assert plan_from_findings(pick, [unknown]).replace == {FieldRef(RETURN)}


def test_rewrite_scope_guard_catches_wholesale_rewrite_but_not_reword():
    from docdebt.repair.verifier import rewrite_scope_issue

    original = (
        "Return the byte array JSON object for the key in this component. Like getValueOr for "
        "subcomponents. This will grab the JSON object for a subcomponent. If the key is not "
        "found or is not a JSON object, a blank subcomponent is returned."
    )
    claim = (
        "This will grab the JSON object for a subcomponent. If the key is not found or is not "
        "a JSON object, a blank subcomponent is returned."
    )
    wholesale_rewrite = (
        "Returns the raw subcomponent as a byte array for the given key. This method parses the "
        "JSON structure and returns the resulting byte array, or an empty subcomponent otherwise."
    )
    assert rewrite_scope_issue(original, claim, wholesale_rewrite) is not None

    minor_reword = (
        "Return the byte array JSON object for the key in this component. Like getValueOr for "
        "subcomponents. This grabs the JSON object for a subcomponent, or an empty one if the "
        "key is not found or is not a JSON object."
    )
    assert rewrite_scope_issue(original, claim, minor_reword) is None

    # the whole field IS the disputed claim: nothing left to preserve, never flagged
    assert (
        rewrite_scope_issue(
            "Returns the maximum of the two values.",
            "Returns the maximum of the two values.",
            "Returns the minimum of the two values.",
        )
        is None
    )


def test_verifier_rejects_a_wholesale_rewrite_of_a_multi_sentence_field():
    unit = units()["toCelsius"]
    original = (
        "/**\n * Converts kelvin to celsius. This method subtracts 273.15 from the input. It is\n"
        " * used throughout the thermal subsystem for display purposes.\n"
        " * @param tempInCelsius temperature in kelvin\n * @return temperature in celsius\n */"
    )
    finding = Finding(
        FindingKind.SEMANTIC,
        Severity.ERROR,
        "c",
        "m",
        unit.id,
        field="description",
        quoted_doc="Converts kelvin to celsius.",
        quoted_code="return tempInKelvin - 273.15;",
    )
    plan = plan_from_findings(unit, [*Auditor().audit(unit).findings, finding])
    rewritten = apply_edits(
        original,
        model_block(
            "Performs a linear transformation on the input to express it on a different scale.",
            "@param tempInKelvin temperature in kelvin",
            "@return temperature in celsius",
        ),
        plan,
        unit.signature,
    )
    issues = Verifier().verify(unit, original, plan, [finding], rewritten)
    assert "rewrite_beyond_claim" in {i.code for i in issues}

    minimal_fix = apply_edits(
        original,
        model_block(
            "Converts kelvin to celsius. This method subtracts 273.15 from the input. It is used "
            "throughout the thermal subsystem for display purposes.",
            "@param tempInKelvin temperature in kelvin",
            "@return temperature in celsius",
        ),
        plan,
        unit.signature,
    )
    issues = Verifier().verify(unit, original, plan, [finding], minimal_fix)
    assert "rewrite_beyond_claim" not in {i.code for i in issues}


def test_load_settings_defaults_base_dir_to_the_analysed_path_not_cwd(tmp_path: Path, monkeypatch):
    from docdebt.settings import load_settings

    monkeypatch.chdir(tmp_path)  # cwd deliberately different from where the code lives
    project = tmp_path / "elsewhere" / "project"
    project.mkdir(parents=True)
    settings = load_settings(search_from=project / "Temperature.java")
    assert settings.base_dir == str(project)
    assert settings.resolve("x.json") == project / "x.json"
