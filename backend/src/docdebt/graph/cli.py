"""Synchronous CLI runner for the graph: ``docdebt review``.

Runs the graph for every method with debt, printing the diff and asking
"accept? [y/N]" on the terminal for each one that reaches a verified repair.
This is the local, single-developer way to exercise the human-review pause;
T3 replaces it with an asynchronous pause resumed from a GitHub PR comment,
using the same graph and the same resume value ("accept" / "reject").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import typer

from docdebt.core.models import MethodUnit
from docdebt.graph.build import GraphApp, GraphMode, build_graph_deps, default_checkpoint_path, sqlite_checkpointer
from docdebt.graph.state import FinalOutcome
from docdebt.repair.apply import apply_repairs_to_file
from docdebt.repair.repairer import RepairResult, RepairStatus
from docdebt.settings import Settings

_OUTCOME_LABEL = {
    FinalOutcome.NO_DEBT.value: "no documentation debt found",
    FinalOutcome.REPAIR_NOT_PROPOSED.value: "debt found, but no verified repair could be proposed",
    FinalOutcome.ACCEPTED.value: "repair accepted",
    FinalOutcome.REJECTED.value: "repair rejected",
}


@dataclass
class ReviewOutcome:
    """One method's result, kept only for the final summary and ``--write``."""

    unit: MethodUnit
    final: str
    repaired_doc: str | None
    changes: list[dict] = field(default_factory=list)

    def to_repair_result(self) -> RepairResult:
        """Adapter to the shape :func:`apply_repairs_to_file` expects."""
        from docdebt.core.javadoc_edit import FieldRef
        from docdebt.repair.repairer import FieldChange

        return RepairResult(
            method_id=self.unit.id,
            status=RepairStatus.VERIFIED,
            original=self.unit.javadoc or "",
            repaired=self.repaired_doc,
            changes=[
                FieldChange(
                    FieldRef(c["field"].split()[0].lstrip("@") or "description", None),
                    c["action"],
                    c["before"],
                    c["after"],
                )
                for c in self.changes
            ],
        )


def _print_diff(unit: MethodUnit, repaired_doc: str) -> None:
    from difflib import unified_diff

    original = (unit.javadoc or "").splitlines()
    typer.echo(
        "\n".join(unified_diff(original, repaired_doc.splitlines(), "current", "proposed", lineterm="", n=2))
    )


def run_review(
    settings: Settings,
    units: list[MethodUnit],
    *,
    mode: GraphMode,
    repair_model: str | None,
    repair_pattern: str | None,
    repair_wording: int | None,
    checkpoint_path: Path | None,
    auto_accept: bool,
    write: bool,
) -> list[ReviewOutcome]:
    """Run the graph for every unit, pausing for a synchronous decision on each
    verified repair. Returns one :class:`ReviewOutcome` per unit."""
    deps = build_graph_deps(
        settings,
        mode=mode,
        repair_model=repair_model,
        repair_pattern=repair_pattern,
        repair_wording=repair_wording,
    )
    path = checkpoint_path or default_checkpoint_path(settings)
    outcomes: list[ReviewOutcome] = []

    with sqlite_checkpointer(path) as saver:
        graph = GraphApp(deps, saver)
        for unit in units:
            thread_id = graph.thread_id(unit)
            state = graph.run_unit(unit, thread_id)
            interrupted = graph.pending_interrupt(thread_id)
            if interrupted is not None:
                typer.echo(f"\n{unit.file_path}:{unit.start_line} {unit.class_name}.{unit.name}")
                _print_diff(unit, interrupted.value["repaired_doc"])
                if interrupted.value["unresolved"]:
                    typer.echo(f"  still to review by hand: {', '.join(interrupted.value['unresolved'])}")
                accept = auto_accept or typer.confirm("  accept this repair?", default=False)
                state = graph.resume(thread_id, "accept" if accept else "reject")
            else:
                label = _OUTCOME_LABEL.get(state.get("final", ""), state.get("final", "?"))
                typer.echo(f"{unit.file_path}:{unit.start_line} {unit.class_name}.{unit.name} -> {label}")
            outcomes.append(
                ReviewOutcome(
                    unit=unit,
                    final=state.get("final", ""),
                    repaired_doc=state.get("repaired_doc"),
                    changes=state.get("changes", []),
                )
            )

    if write:
        _write_accepted(outcomes)
    return outcomes


def _write_accepted(outcomes: list[ReviewOutcome]) -> None:
    by_file: dict[str, list[tuple[MethodUnit, RepairResult]]] = {}
    for outcome in outcomes:
        if outcome.final == FinalOutcome.ACCEPTED.value:
            by_file.setdefault(outcome.unit.file_path, []).append((outcome.unit, outcome.to_repair_result()))
    total = 0
    for file_path, items in by_file.items():
        written, skipped = apply_repairs_to_file(file_path, items)
        total += written
        for message in skipped:
            typer.echo(f"skipped: {message}", err=True)
    typer.echo(f"\n{total} accepted repair(s) written.")
