"""Command-line interface: ``docdebt check`` (structural) and ``docdebt detect`` (with LLM)."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import typer

from docdebt.core.auditor import Auditor
from docdebt.core.extractor import Extractor
from docdebt.core.models import Finding, MethodUnit, Severity
from docdebt.settings import Settings, load_settings

app = typer.Typer(add_completion=False, help="Detect stale method-level Javadoc in Java code.")


class OutputFormat(StrEnum):
    TEXT = "text"
    JSON = "json"


class FailOn(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    NEVER = "never"


def _load(config: Path | None, anchor: Path) -> Settings:
    return load_settings(config, search_from=anchor)


def _collect_units(
    settings: Settings, paths: list[Path], diff: Path | None, root: Path | None
) -> list[MethodUnit]:
    extractor = Extractor(settings.extractor.to_config())
    if diff is not None:
        if root is None:
            raise typer.BadParameter("--diff requires --root (the checked-out new version).")
        return extractor.from_diff(diff.read_text(encoding="utf-8"), root)
    if not paths:
        raise typer.BadParameter("Give at least one file or directory, or --diff with --root.")
    try:
        return extractor.from_paths(paths)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _format_finding(unit: MethodUnit, finding: Finding) -> str:
    where = f"{unit.file_path}:{finding.line or unit.start_line}"
    text = f"{where}: {finding.severity.value} [{finding.rule}] {finding.message} ({unit.class_name}.{unit.name})"
    if finding.quoted_doc:
        text += f'\n    doc:  "{finding.quoted_doc}"\n    code: "{finding.quoted_code}"'
        if finding.confidence is not None:
            text += f"\n    confidence: {finding.confidence:.2f} ({finding.model})"
    return text


@app.command()
def check(
    paths: list[Path] = typer.Argument(None, help="Java files or directories."),
    diff: Path | None = typer.Option(None, help="Unified diff; only touched methods are checked."),
    root: Path | None = typer.Option(None, help="Repository root for --diff."),
    config: Path | None = typer.Option(None, help="Configuration file (default: nearest .docdebt.yml)."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", help="Output format."),
    fail_on: FailOn = typer.Option(FailOn.ERROR, help="Exit with code 1 on this severity."),
) -> None:
    """Deterministic structural check (no LLM): @param, @return, @throws against the signature."""
    anchor = root or (paths[0] if paths else Path.cwd())
    settings = _load(config, anchor)
    units = _collect_units(settings, paths or [], diff, root)
    reports = Auditor(settings.auditor.to_config()).audit_all(units)

    if output == OutputFormat.JSON:
        typer.echo(
            json.dumps(
                [{"method_id": r.unit.id, "findings": [f.to_dict() for f in r.findings]} for r in reports],
                indent=2,
            )
        )
    else:
        for report in reports:
            for finding in report.findings:
                typer.echo(_format_finding(report.unit, finding))
        n_err = sum(len(r.errors) for r in reports)
        n_warn = sum(len(r.warnings) for r in reports)
        n_debt = sum(r.has_structural_debt for r in reports)
        typer.echo(
            f"\n{len(reports)} method(s) checked, {n_debt} with structural debt "
            f"({n_err} error(s), {n_warn} warning(s))."
        )

    failing = {
        FailOn.ERROR: any(r.errors for r in reports),
        FailOn.WARNING: any(r.findings for r in reports),
        FailOn.NEVER: False,
    }[fail_on]
    raise typer.Exit(code=1 if failing else 0)


@app.command()
def detect(
    paths: list[Path] = typer.Argument(None, help="Java files or directories."),
    diff: Path | None = typer.Option(None, help="Unified diff; only touched methods are analysed."),
    root: Path | None = typer.Option(None, help="Repository root (for --diff and for context)."),
    config: Path | None = typer.Option(None, help="Configuration file (default: nearest .docdebt.yml)."),
    model: list[str] = typer.Option(None, help="Override the cascade with these model ids, in order."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", help="Output format."),
    show_votes: bool = typer.Option(False, help="Print every raw model answer."),
    fail_on_debt: bool = typer.Option(True, help="Exit with code 1 when debt is found."),
) -> None:
    """Structural check plus LLM semantic detection with validated citations."""
    from docdebt.pipeline import build_detection_pipeline

    anchor = root or (paths[0] if paths else Path.cwd())
    settings = _load(config, anchor)
    units = _collect_units(settings, paths or [], diff, root)
    pipeline = build_detection_pipeline(settings, repo_root=root, cascade=model or None)
    results = pipeline.run(units)

    if output == OutputFormat.JSON:
        payload = []
        for r in results:
            payload.append(
                {
                    "method_id": r.unit.id,
                    "status": r.status.value,
                    "structural_debt": r.has_structural_debt,
                    "semantic_status": r.semantic_status.value,
                    "confidence": r.final.confidence if r.final else None,
                    "reason": r.final.reason if r.final else r.note,
                    "models": [t.model_id for t in r.trail],
                    "findings": [f.to_dict() for f in r.findings],
                    **(
                        {
                            "votes": [
                                {
                                    "model": t.model_id,
                                    "pattern": v.pattern,
                                    "verdict": v.verdict,
                                    "raw": v.raw,
                                }
                                for t in r.trail
                                for v in t.votes
                            ]
                        }
                        if show_votes
                        else {}
                    ),
                }
            )
        typer.echo(json.dumps(payload, indent=2))
    else:
        for r in results:
            source = "structural" if r.has_structural_debt else "llm"
            head = f"{r.unit.file_path}:{r.unit.start_line} {r.unit.class_name}.{r.unit.name} -> {r.status.value}"
            if r.status.value == "debt":
                head += f" [{source}]"
            if r.final:
                head += (
                    f" | LLM {r.final.model_id}: {r.final.status.value}, "
                    f"confidence {r.final.confidence:.2f}, {r.final.reason}"
                )
            elif r.note:
                head += f" | {r.note}"
            typer.echo(head)
            for f in r.findings:
                if f.severity == Severity.ERROR:
                    typer.echo("  " + _format_finding(r.unit, f).replace("\n", "\n  "))
            if show_votes:
                for t in r.trail:
                    for v in t.votes:
                        typer.echo(f"    [{t.model_id} {v.pattern}#{v.sample}] -> {v.verdict}: {v.raw!r}")
        usage = pipeline.tracker
        typer.echo(
            f"\n{len(results)} method(s); {usage.calls} LLM call(s), "
            f"{usage.total_tokens} tokens, ${usage.cost_usd:.4f}."
        )

    has_debt = any(r.status.value == "debt" for r in results)
    raise typer.Exit(code=1 if (fail_on_debt and has_debt) else 0)


@app.command()
def repair(
    paths: list[Path] = typer.Argument(None, help="Java files or directories."),
    diff: Path | None = typer.Option(None, help="Unified diff; only touched methods are repaired."),
    root: Path | None = typer.Option(None, help="Repository root (for --diff and for context)."),
    config: Path | None = typer.Option(None, help="Configuration file (default: nearest .docdebt.yml)."),
    model: str | None = typer.Option(
        None, help="Repair model id (default: repair.model, then the first cascade model)."
    ),
    pattern: str | None = typer.Option(None, help="Repair prompt pattern: zeroshot, oneshot, threeshot, cot."),
    wording: int | None = typer.Option(
        None, help="Wording index, required for models not screened in the research."
    ),
    semantic: bool = typer.Option(False, help="Also detect semantic debt with the Judge (needs a model)."),
    hint_findings: bool = typer.Option(
        False,
        "--hint-findings",
        help="Tell the repair model which problems were found (cited claim, missing tags).",
    ),
    write: bool = typer.Option(False, help="Write the verified repairs into the source files."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", help="Output format."),
) -> None:
    """Propose a verified repair for every method with documentation debt.

    Only the fields that must change are edited; everything else stays byte for byte.
    Nothing is written to disk unless --write is given.
    """
    from docdebt.pipeline import build_detection_pipeline, build_repairer
    from docdebt.repair.apply import apply_repairs_to_file, javadoc_diff
    from docdebt.repair.repairer import RepairStatus

    anchor = root or (paths[0] if paths else Path.cwd())
    settings = _load(config, anchor)
    units = _collect_units(settings, paths or [], diff, root)
    repairer = build_repairer(
        settings, model, pattern=pattern, wording=wording, hint_findings=True if hint_findings else None
    )
    tracker = repairer.client.tracker

    if semantic:
        pipeline = build_detection_pipeline(settings, repo_root=root)
        findings_by_unit = [(r.unit, r.findings if r.status.value == "debt" else []) for r in pipeline.run(units)]
        usage = pipeline.tracker
        tracker.calls += usage.calls
        tracker.input_tokens += usage.input_tokens
        tracker.output_tokens += usage.output_tokens
        tracker.cost_usd += usage.cost_usd
    else:
        auditor = Auditor(settings.auditor.to_config())
        findings_by_unit = [(u, auditor.audit(u).findings) for u in units]

    outcomes = []
    for unit, findings in findings_by_unit:
        if any(f.is_debt for f in findings):
            outcomes.append((unit, repairer.repair(unit, findings)))

    if output == OutputFormat.JSON:
        typer.echo(
            json.dumps(
                [
                    {
                        "method_id": u.id,
                        "status": r.status.value,
                        "reason": r.reason,
                        "attempts": len(r.attempts),
                        "changes": [
                            {"field": c.ref.label, "action": c.action, "before": c.before, "after": c.after}
                            for c in r.changes
                        ],
                        "unresolved": r.unresolved,
                        "repaired": r.repaired,
                    }
                    for u, r in outcomes
                ],
                indent=2,
            )
        )
    else:
        for unit, result in outcomes:
            typer.echo(
                f"{unit.file_path}:{unit.start_line} {unit.class_name}.{unit.name} -> {result.status.value}"
                + (f" ({result.reason})" if result.reason else "")
                + f" [{len(result.attempts)} attempt(s), model {result.model_id}]"
            )
            if result.status == RepairStatus.VERIFIED:
                typer.echo(javadoc_diff(unit, result))
                if result.unresolved:
                    typer.echo(f"  still to review by hand: {', '.join(result.unresolved)}")
        counts = {s: sum(r.status == s for _, r in outcomes) for s in RepairStatus}
        typer.echo(
            f"\n{len(units)} method(s) analysed, {len(outcomes)} with debt: "
            + ", ".join(f"{n} {s.value}" for s, n in counts.items() if n)
        )
        typer.echo(f"{tracker.calls} LLM call(s), {tracker.total_tokens} tokens, ${tracker.cost_usd:.4f}.")

    if write:
        by_file: dict[str, list] = {}
        for unit, result in outcomes:
            by_file.setdefault(unit.file_path, []).append((unit, result))
        total = 0
        for file_path, items in by_file.items():
            written, skipped = apply_repairs_to_file(root / file_path if root else file_path, items)
            total += written
            for message in skipped:
                typer.echo(f"skipped: {message}", err=True)
        typer.echo(f"{total} Javadoc(s) written.", err=output == OutputFormat.JSON)


@app.command()
def review(
    paths: list[Path] = typer.Argument(None, help="Java files or directories."),
    diff: Path | None = typer.Option(None, help="Unified diff; only touched methods are reviewed."),
    root: Path | None = typer.Option(None, help="Repository root (for --diff, context, and --write)."),
    config: Path | None = typer.Option(None, help="Configuration file (default: nearest .docdebt.yml)."),
    model: str | None = typer.Option(
        None, help="Repair model id (default: repair.model, then the first cascade model)."
    ),
    pattern: str | None = typer.Option(None, help="Repair prompt pattern: zeroshot, oneshot, threeshot, cot."),
    wording: int | None = typer.Option(
        None, help="Wording index, required for models not screened in the research."
    ),
    structural_only: bool = typer.Option(
        False, "--structural-only", help="Skip the Judge: repair structural debt only, no LLM detection call."
    ),
    checkpoint_db: Path | None = typer.Option(
        None, help="SQLite checkpoint file (default: <config dir>/.docdebt/graph.sqlite3)."
    ),
    yes: bool = typer.Option(False, "--yes", help="Accept every verified repair without asking."),
    write: bool = typer.Option(False, "--write", help="Write accepted repairs into the source files."),
) -> None:
    """Detect and repair documentation debt one method at a time, with a review pause.

    Runs the graph: audit, detect (unless --structural-only), and, for a
    verified repair, a pause where you accept or reject it before anything is
    written. State is checkpointed after every step, so interrupting this
    command and re-running it later resumes exactly where it stopped,
    including a run still waiting on your decision — resuming re-invokes this
    same command; a thread already paused or finished is not re-analysed.
    """
    from docdebt.graph.cli import run_review

    anchor = root or (paths[0] if paths else Path.cwd())
    settings = _load(config, anchor)
    units = _collect_units(settings, paths or [], diff, root)
    run_review(
        settings,
        units,
        mode="structural" if structural_only else "pipeline",
        repair_model=model,
        repair_pattern=pattern,
        repair_wording=wording,
        checkpoint_path=checkpoint_db,
        auto_accept=yes,
        write=write,
    )


if __name__ == "__main__":
    app()
