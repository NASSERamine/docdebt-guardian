from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from docdebt.core.auditor import Auditor
from docdebt.core.java_parser import parse_java_source
from docdebt.graph.build import (
    GraphApp,
    build_graph_deps,
    default_checkpoint_path,
    sqlite_checkpointer,
)
from docdebt.graph.state import FinalOutcome, ReviewDecision
from docdebt.llm.client import LLMClient
from docdebt.repair.registry import RepairRegistry
from docdebt.repair.repairer import RepairConfig, Repairer, RepairStatus
from docdebt.settings import Settings

from .conftest import KELVIN_SOURCE, make_factory

GOOD_CITATION = (
    "CLAIM: Returns the maximum of the two values.\n"
    "CODE: return Math.min(a, b);\n"
    "REASON: it returns the minimum\nFIELD: description\n---"
)


def units():
    return {u.name: u for u in parse_java_source(KELVIN_SOURCE, "demo/Temperature.java")}


def is_explain(prompt: str) -> bool:
    return "Find every such contradicted claim" in prompt


def deps_with(local_spec, responder, *, mode="pipeline"):
    from docdebt.graph.nodes import GraphDeps

    factory = make_factory(responder)
    if mode == "pipeline":
        from docdebt.detection.judge import Judge
        from docdebt.detection.router import Router, RouterPolicy
        from docdebt.llm.prompts.registry import PromptRegistry

        judge = Judge(LLMClient(local_spec, factory=factory, sleep=lambda s: None), PromptRegistry())
        router = Router({local_spec.id: judge}, RouterPolicy(cascade=[local_spec.id]))
    else:
        router = None
    repairer = Repairer(
        LLMClient(local_spec, factory=factory, sleep=lambda s: None), RepairRegistry(), config=RepairConfig()
    )
    return GraphDeps(auditor=Auditor(), router=router, repairer=repairer)


def app_with(deps) -> GraphApp:
    return GraphApp(deps, InMemorySaver())


def test_no_debt_completes_without_pausing(local_spec):
    clean = units()["pick"]  # no structural debt; force a "clean" judge answer

    def responder(prompt, m):
        return ("NONE" if is_explain(prompt) else "COMPLETE"), "stop"

    app = app_with(deps_with(local_spec, responder))
    state = app.run_unit(clean, thread_id="clean-1")
    assert state["final"] == FinalOutcome.NO_DEBT.value
    assert app.pending_interrupt("clean-1") is None


def structural_repair_responder(prompt: str, _max_tokens: int) -> tuple[str, str]:
    """A valid Javadoc for the repair prompt; irrelevant for every other prompt
    in a structural-only graph (no Judge, so no vote or citation prompt is sent)."""
    text = (
        "/**\n * Converts kelvin to celsius.\n * @param tempInKelvin temperature in kelvin\n"
        " * @return temperature in celsius\n */"
        if "CURRENT DOCUMENTATION:" in prompt
        else "NONE"
    )
    return text, "stop"


def test_structural_debt_pauses_then_accept_or_reject(local_spec):
    unit = units()["toCelsius"]
    app = app_with(deps_with(local_spec, structural_repair_responder, mode="structural"))

    paused = app.run_unit(unit, thread_id="accept-me")
    assert "final" not in paused  # still waiting on review
    interrupt = app.pending_interrupt("accept-me")
    assert interrupt is not None and "tempInKelvin temperature in kelvin" in interrupt.value["repaired_doc"]
    final = app.resume("accept-me", ReviewDecision.ACCEPT.value)
    assert final["final"] == FinalOutcome.ACCEPTED.value
    assert final["review_decision"] == ReviewDecision.ACCEPT.value

    app2 = app_with(deps_with(local_spec, structural_repair_responder, mode="structural"))
    app2.run_unit(unit, thread_id="reject-me")
    final2 = app2.resume("reject-me", ReviewDecision.REJECT.value)
    assert final2["final"] == FinalOutcome.REJECTED.value
    # a rejected repair is never lost: it is still visible for inspection
    assert final2["repaired_doc"] is not None


def test_semantic_debt_with_cited_contradiction_pauses(local_spec):
    unit = units()["pick"]

    def responder(prompt, m):
        if is_explain(prompt):
            return GOOD_CITATION, "stop"
        return ("INCOMPLETE" if "MODEL_REPAIR_MARKER" not in prompt else "COMPLETE"), "stop"

    # the repair call itself must return a fixed Javadoc
    def responder_with_repair(prompt, m):
        if "CURRENT DOCUMENTATION:" in prompt:
            return "/**\n * Returns the minimum of the two values.\n */", "stop"
        return responder(prompt, m)

    app = app_with(deps_with(local_spec, responder_with_repair))
    state = app.run_unit(unit, thread_id="semantic-1")
    assert state["detection_status"] == "debt"
    assert state["repair_status"] == RepairStatus.VERIFIED.value
    assert app.pending_interrupt("semantic-1") is not None
    final = app.resume("semantic-1", "accept")
    assert final["final"] == FinalOutcome.ACCEPTED.value
    assert "minimum" in final["repaired_doc"]


def test_repair_failure_completes_without_pausing(local_spec):
    unit = units()["read"]  # missing @throws; give the model an answer missing the tag
    app = app_with(deps_with(local_spec, lambda p, m: ("no javadoc block here", "stop"), mode="structural"))
    state = app.run_unit(unit, thread_id="fail-1")
    assert state["final"] == FinalOutcome.REPAIR_NOT_PROPOSED.value
    assert state["repair_status"] == RepairStatus.FAILED.value
    assert app.pending_interrupt("fail-1") is None


def test_checkpoint_survives_a_simulated_process_restart(local_spec, tmp_path: Path):
    db = tmp_path / "graph.sqlite3"
    unit = units()["toCelsius"]

    with sqlite_checkpointer(db) as saver:
        app1 = GraphApp(deps_with(local_spec, structural_repair_responder, mode="structural"), saver)
        app1.run_unit(unit, thread_id="restart-me")
        assert app1.pending_interrupt("restart-me") is not None
    # app1, its checkpointer connection and its in-process deps are gone here,
    # simulating the CLI process having exited between "run" and "resume"

    with sqlite_checkpointer(db) as saver2:
        app2 = GraphApp(deps_with(local_spec, pytest_fail_if_called, mode="structural"), saver2)
        assert app2.pending_interrupt("restart-me") is not None  # still paused, read from disk
        final = app2.resume("restart-me", "accept")
    assert final["final"] == FinalOutcome.ACCEPTED.value


def pytest_fail_if_called(*_a, **_k) -> tuple[str, str]:
    import pytest

    pytest.fail("a fresh GraphApp must not need to re-run earlier nodes to resume")
    return "", "stop"


def test_thread_id_is_stable_and_default_checkpoint_path_is_under_docdebt_dir():
    unit = units()["pick"]
    assert GraphApp.thread_id(unit) == unit.id
    path = default_checkpoint_path(Settings(base_dir="/repo"))
    assert path == Path("/repo/.docdebt/graph.sqlite3")


def test_build_graph_deps_structural_mode_has_no_router(local_spec, monkeypatch):
    monkeypatch.setattr(
        "docdebt.llm.client.default_chat_model_factory", make_factory(lambda p, m: ("NONE", "stop"))
    )
    settings = Settings(router={"cascade": [local_spec.id]}, repair={"model": local_spec.id})
    deps = build_graph_deps(settings, mode="structural")
    assert deps.router is None
    deps_pipeline = build_graph_deps(settings, mode="pipeline")
    assert deps_pipeline.router is not None


def test_run_unit_never_reinvokes_over_a_pending_review(local_spec):
    """A second call to run_unit on an already-paused thread must not repeat any
    LLM call: doing so would both waste money and risk showing the reviewer a
    different repair than the one they are about to answer for (models are not
    perfectly deterministic in practice)."""
    calls: list[str] = []

    def counting_responder(prompt, m):
        calls.append(prompt)
        return structural_repair_responder(prompt, m)

    unit = units()["toCelsius"]
    app = app_with(deps_with(local_spec, counting_responder, mode="structural"))

    app.run_unit(unit, thread_id="no-double-spend")
    first_call_count = len(calls)
    assert first_call_count > 0

    again = app.run_unit(unit, thread_id="no-double-spend")
    assert len(calls) == first_call_count  # not called again
    assert "final" not in again  # still paused, not silently finalized either


def test_run_unit_reuses_a_finished_thread_for_unchanged_content(local_spec):
    calls: list[str] = []

    def counting_responder(prompt, m):
        calls.append(prompt)
        return structural_repair_responder(prompt, m)

    unit = units()["toCelsius"]
    app = app_with(deps_with(local_spec, counting_responder, mode="structural"))
    app.run_unit(unit, thread_id="reuse-me")
    app.resume("reuse-me", "accept")
    count_after_finish = len(calls)

    app.run_unit(unit, thread_id="reuse-me")  # same content, same thread: must be free
    assert len(calls) == count_after_finish


def test_run_unit_reanalyses_edited_content_under_the_same_thread_id(local_spec):
    import dataclasses

    calls: list[str] = []

    def counting_responder(prompt, m):
        calls.append(prompt)
        return structural_repair_responder(prompt, m)

    unit = units()["toCelsius"]
    app = app_with(deps_with(local_spec, counting_responder, mode="structural"))
    app.run_unit(unit, thread_id="edited")
    app.resume("edited", "accept")
    count_after_first = len(calls)

    edited = dataclasses.replace(unit, javadoc="/** A different Javadoc entirely. */")
    app.run_unit(edited, thread_id="edited")
    assert len(calls) > count_after_first
    assert app.pending_interrupt("edited") is not None
