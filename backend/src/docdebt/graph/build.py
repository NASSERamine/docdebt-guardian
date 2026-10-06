"""Compile the graph and expose a small, synchronous API to run and resume it.

One compiled graph is built once per process (per repair configuration) and
reused for every method: ``run_unit`` starts a fresh thread for one method,
``resume`` continues a paused one. Both are synchronous, matching the
current CLI-driven, one-reviewer-at-a-time workflow; an async/webhook-driven
resume (for the GitHub App) is a later phase and will call the same compiled
graph through ``ainvoke`` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from docdebt.core.auditor import Auditor
from docdebt.core.models import MethodUnit
from docdebt.core.serialize import unit_to_dict
from docdebt.detection.context import ContextRetriever
from docdebt.detection.judge import Judge
from docdebt.detection.router import Router
from docdebt.graph.nodes import (
    GraphDeps,
    content_fingerprint,
    finalize_no_debt,
    finalize_not_proposed,
    has_debt,
    is_verified,
    make_audit_node,
    make_detect_node,
    make_repair_node,
    review,
)
from docdebt.graph.state import DocDebtState
from docdebt.llm.catalog import get_model_spec
from docdebt.llm.client import ChatModelFactory, LLMClient, UsageTracker
from docdebt.pipeline import build_registry
from docdebt.settings import Settings

GraphMode = Literal["structural", "pipeline"]
DEFAULT_CHECKPOINT_NAME = "graph.sqlite3"


def _build_router(
    settings: Settings, repo_root: str | Path | None, factory: ChatModelFactory | None, tracker: UsageTracker
) -> Router | None:
    policy = settings.router.to_policy()
    if not policy.cascade:
        return None
    registry = build_registry(settings)
    specs = settings.model_specs()
    judges = {
        model_id: Judge(
            LLMClient(get_model_spec(model_id, specs), factory=factory, tracker=tracker),
            registry,
            settings.judge_config(model_id),
        )
        for model_id in policy.cascade
    }
    context_provider = None
    if any(j.config.use_context for j in judges.values()) and repo_root is not None:
        context_provider = ContextRetriever(repo_root, settings.context.sources)
    return Router(judges, policy, tracker, context_provider)


def build_graph_deps(
    settings: Settings,
    *,
    mode: GraphMode = "pipeline",
    repo_root: str | Path | None = None,
    repair_model: str | None = None,
    repair_pattern: str | None = None,
    repair_wording: int | None = None,
    factory: ChatModelFactory | None = None,
    tracker: UsageTracker | None = None,
) -> GraphDeps:
    """Build the components the graph will call, exactly as the CLI's ``detect``
    and ``repair`` commands do, so the graph behaves identically to them."""
    from docdebt.pipeline import build_repairer  # local import: avoids a build.py <-> pipeline.py cycle

    tracker = tracker or UsageTracker()
    router = _build_router(settings, repo_root, factory, tracker) if mode == "pipeline" else None
    repairer = build_repairer(
        settings, repair_model, pattern=repair_pattern, wording=repair_wording, factory=factory, tracker=tracker
    )
    return GraphDeps(auditor=Auditor(settings.auditor.to_config()), router=router, repairer=repairer)


def compile_graph(deps: GraphDeps, checkpointer: BaseCheckpointSaver):
    """Wire the nodes into the graph described in the architecture: Auditor,
    then Router/Judge, then Repairer, then a human-review pause before
    anything is proposed as final."""
    graph = StateGraph(DocDebtState)
    graph.add_node("audit", make_audit_node(deps))
    graph.add_node("detect", make_detect_node(deps))
    graph.add_node("repair", make_repair_node(deps))
    graph.add_node("review", review)
    # mypy misreads these two plain, explicitly-typed node functions against
    # langgraph's StateGraph stub (a contravariance quirk in langgraph 1.2's
    # typing); the closures returned by make_*_node above have the identical
    # runtime shape and are not flagged, only because their return type is
    # inferred rather than annotated.
    graph.add_node("finalize_no_debt", finalize_no_debt)  # type: ignore[arg-type]
    graph.add_node("finalize_not_proposed", finalize_not_proposed)  # type: ignore[arg-type]

    graph.add_edge(START, "audit")
    graph.add_edge("audit", "detect")
    graph.add_conditional_edges("detect", has_debt, {True: "repair", False: "finalize_no_debt"})
    graph.add_conditional_edges("repair", is_verified, {True: "review", False: "finalize_not_proposed"})
    graph.add_edge("review", END)
    graph.add_edge("finalize_no_debt", END)
    graph.add_edge("finalize_not_proposed", END)
    return graph.compile(checkpointer=checkpointer)


@contextmanager
def sqlite_checkpointer(path: str | Path) -> Iterator[BaseCheckpointSaver]:
    """A SQLite checkpointer at ``path``, creating parent directories as needed.
    Persists across process restarts, unlike the default in-memory checkpointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def default_checkpoint_path(settings: Settings) -> Path:
    return Path(settings.base_dir) / ".docdebt" / DEFAULT_CHECKPOINT_NAME


class GraphApp:
    """A compiled graph bound to one checkpointer, with a synchronous run/resume API."""

    def __init__(self, deps: GraphDeps, checkpointer: BaseCheckpointSaver) -> None:
        self.deps = deps
        self.checkpointer = checkpointer
        self.compiled = compile_graph(deps, checkpointer)

    @staticmethod
    def thread_id(unit: MethodUnit) -> str:
        """Stable id for a method's graph thread: reusing it resumes the same run."""
        return unit.id

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def run_unit(self, unit: MethodUnit, thread_id: str | None = None) -> DocDebtState:
        """Reattach to ``thread_id`` if it exists, or start a fresh run for ``unit``.

        Safe to call every time the CLI runs, including once per method on
        every re-invocation of ``docdebt review``: an in-progress run is
        never invoked again (a thread paused at the review step is returned
        as-is — calling this would otherwise silently repeat every LLM call
        made so far, and Ollama is not perfectly deterministic, so a second
        call could even change the very repair the person is about to
        approve). A *finished* run for content that has not changed since
        (same code and Javadoc, via :func:`~docdebt.graph.nodes.content_fingerprint`)
        is also reused rather than re-analysed, so scanning a whole
        repository repeatedly does not re-pay for methods nobody touched.
        Only unseen content — a new thread, or a finished thread whose
        method was edited since — starts a fresh run.
        """
        thread_id = thread_id or self.thread_id(unit)
        snapshot = self.compiled.get_state(self._config(thread_id))
        if snapshot.next:  # paused or otherwise mid-run: never re-invoke over it
            return snapshot.values
        if snapshot.values and snapshot.values.get("content_fingerprint") == content_fingerprint(unit):
            return snapshot.values  # already finished for this exact content

        initial: DocDebtState = {
            "unit": unit_to_dict(unit),
            "structural_findings": [],
            "has_structural_debt": False,
        }
        self.compiled.invoke(initial, self._config(thread_id))
        return self.get_state(thread_id)

    def resume(self, thread_id: str, decision: str) -> DocDebtState:
        """Continue a paused thread with a human decision ("accept" or "reject")."""
        self.compiled.invoke(Command(resume=decision), self._config(thread_id))
        return self.get_state(thread_id)

    def get_state(self, thread_id: str) -> DocDebtState:
        return self.compiled.get_state(self._config(thread_id)).values

    def pending_interrupt(self, thread_id: str) -> Interrupt | None:
        """The paused review payload for ``thread_id``, or ``None`` if it is not paused."""
        snapshot = self.compiled.get_state(self._config(thread_id))
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0]
        return None
