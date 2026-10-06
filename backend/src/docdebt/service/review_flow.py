"""Orchestration: a pull request event in, PR comments (and, on accept, a
commit) out. Everything Java-specific or LLM-specific is delegated to the
already-validated core (``docdebt.core``, ``docdebt.graph``,
``docdebt.repair``) — this module only adapts them to a GitHub App, which has
no local checkout and must talk to files through the REST API instead of the
filesystem.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from docdebt.core.diff import parse_unified_diff
from docdebt.core.extractor import Extractor
from docdebt.core.models import MethodUnit
from docdebt.graph.build import GraphApp
from docdebt.graph.state import DocDebtState, FinalOutcome
from docdebt.llm.client import UsageTracker
from docdebt.repair.apply import apply_repairs_to_text
from docdebt.repair.repairer import FieldChange, RepairResult, RepairStatus
from docdebt.service.commands import SlashCommand
from docdebt.service.github_client import GithubApiError, GithubClient
from docdebt.service.store import ReviewRequest, ReviewStore
from docdebt.service.webhooks import CommentEvent, PullRequestEvent

logger = logging.getLogger("docdebt.service")

_MARKER = "<!-- docdebt:managed-comment -->"


def _method_label(unit: MethodUnit) -> str:
    return f"{unit.class_name}.{unit.name}"


def extract_touched_units(diff_text: str, file_contents: dict[str, str], extractor: Extractor) -> list[MethodUnit]:
    """Methods touched by ``diff_text``, parsed from ``file_contents`` (path ->
    full file text at the PR's head) fetched over the GitHub API.

    Mirrors :meth:`docdebt.core.extractor.Extractor.from_diff`, which instead
    reads a local checkout; there is no local checkout here, only whatever
    files the caller already fetched.
    """
    units: list[MethodUnit] = []
    for path, change in parse_unified_diff(diff_text).items():
        if not path.endswith(".java") or extractor.is_excluded(path) or path not in file_contents:
            continue
        for unit in extractor.from_source(file_contents[path], path):
            if change.touches(unit.span_start, unit.end_line):
                units.append(unit)
    return units


def _proposal_body(unit: MethodUnit, repaired_doc: str, changes: list[dict], short_id: str) -> str:
    from difflib import unified_diff

    diff = "\n".join(
        unified_diff((unit.javadoc or "").splitlines(), repaired_doc.splitlines(), "current", "proposed", n=2)
    )
    fields = ", ".join(c["field"] for c in changes) or "the documentation"
    return (
        f"{_MARKER}\n"
        f"### 📝 Documentation debt in `{unit.file_path}:{unit.start_line}` — "
        f"`{_method_label(unit)}`\n\n"
        f"Proposed fix for {fields}:\n\n"
        f"```diff\n{diff}\n```\n\n"
        f"Reply `/docdebt accept {short_id}` to apply this to the branch, or "
        f"`/docdebt reject {short_id}` to dismiss it."
    )


@dataclass
class ReviewOutcome:
    proposed: int
    skipped_no_debt: int
    skipped_not_proposed: int


class ReviewFlow:
    """Bound to one installation's GitHub client, the compiled graph, and the
    short-id store. One instance handles every PR the service is watching."""

    def __init__(
        self, github: GithubClient, graph: GraphApp, store: ReviewStore, tracker: UsageTracker | None = None
    ) -> None:
        self.github = github
        self.graph = graph
        self.store = store
        self.extractor = Extractor()
        # Passed in by the caller (docdebt.service.app) as the SAME tracker given to
        # build_graph_deps, so its totals reflect every LLM call the graph actually
        # made -- not a tracker of its own that nothing ever updates.
        self.tracker = tracker or UsageTracker()
        self._last_usage = (
            self.tracker.calls,
            self.tracker.input_tokens,
            self.tracker.output_tokens,
            self.tracker.cost_usd,
        )

    def _log_usage(self, kind: str, owner: str, repo: str, pr_number: int, latency_s: float) -> None:
        """Log the shared tracker's growth since the previous call, as one row.

        The tracker accumulates for the whole process, so only the *delta*
        since the last log is stored (``self._last_usage`` is the running
        baseline); this makes it safe to call after every webhook even
        though one process, and its tracker, outlives many of them.
        """
        before = self._last_usage
        self._last_usage = (
            self.tracker.calls,
            self.tracker.input_tokens,
            self.tracker.output_tokens,
            self.tracker.cost_usd,
        )
        self.store.log_usage(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            kind=kind,
            calls=self.tracker.calls - before[0],
            input_tokens=self.tracker.input_tokens - before[1],
            output_tokens=self.tracker.output_tokens - before[2],
            cost_usd=self.tracker.cost_usd - before[3],
            latency_s=latency_s,
        )

    # -- pull_request: opened / synchronize -----------------------------------

    def review_pull_request(self, event: PullRequestEvent) -> ReviewOutcome:
        started = time.perf_counter()
        diff = self.github.get_pull_request_diff(event.owner, event.repo, event.number)
        needed_paths = {
            path
            for path in parse_unified_diff(diff)
            if path.endswith(".java") and not self.extractor.is_excluded(path)
        }
        file_contents = {
            path: self.github.get_file_content(event.owner, event.repo, path, event.head_sha)[0]
            for path in needed_paths
        }
        touched = extract_touched_units(diff, file_contents, self.extractor)

        outcome = ReviewOutcome(proposed=0, skipped_no_debt=0, skipped_not_proposed=0)
        for unit in touched:
            thread_id = f"{event.owner}/{event.repo}#{event.number}:{unit.id}"
            state = self.graph.run_unit(unit, thread_id)
            interrupted = self.graph.pending_interrupt(thread_id)
            if interrupted is None:
                if state.get("final") == FinalOutcome.NO_DEBT.value:
                    outcome.skipped_no_debt += 1
                else:
                    outcome.skipped_not_proposed += 1
                continue
            short_id = self.store.create(
                thread_id=thread_id,
                owner=event.owner,
                repo=event.repo,
                pr_number=event.number,
                file_path=unit.file_path,
                method_label=_method_label(unit),
                head_sha=event.head_sha,
            )
            body = _proposal_body(unit, interrupted.value["repaired_doc"], interrupted.value["changes"], short_id)
            self.github.create_issue_comment(event.owner, event.repo, event.number, body)
            outcome.proposed += 1
        logger.info(
            "reviewed %s/%s#%s: %s proposed, %s clean, %s not proposed",
            event.owner,
            event.repo,
            event.number,
            outcome.proposed,
            outcome.skipped_no_debt,
            outcome.skipped_not_proposed,
        )
        self._log_usage("review", event.owner, event.repo, event.number, time.perf_counter() - started)
        return outcome

    # -- issue_comment: created (a slash command reply) -----------------------

    def handle_command(self, event: CommentEvent, command: SlashCommand) -> None:
        started = time.perf_counter()
        self.github.add_reaction(event.owner, event.repo, event.comment_id)
        request = self.store.get(command.short_id)
        if (
            request is None
            or request.resolved
            or (request.owner, request.repo, request.pr_number)
            != (
                event.owner,
                event.repo,
                event.issue_number,
            )
        ):
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.issue_number,
                f"{_MARKER}\n⚠️ No pending review found for id `{command.short_id}` on this pull request "
                "(it may already be resolved, or belong to a different PR).",
            )
            return

        state = self.graph.resume(request.thread_id, command.decision)
        self.store.mark_resolved(command.short_id, command.decision)
        self._log_usage("command", event.owner, event.repo, event.issue_number, time.perf_counter() - started)

        if command.decision != "accept" or state.get("final") != FinalOutcome.ACCEPTED.value:
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.issue_number,
                f"{_MARKER}\n🚫 Repair for `{request.method_label}` rejected — no changes made.",
            )
            return

        self._commit_accepted(event, request, state)

    def _commit_accepted(self, event: CommentEvent, request: ReviewRequest, state: DocDebtState) -> None:
        pr = self.github.get_pull_request(event.owner, event.repo, event.issue_number)
        branch = pr["head"]["ref"]
        content, sha = self.github.get_file_content(event.owner, event.repo, request.file_path, branch)

        unit_for_apply = _minimal_unit_for_apply(state)
        result = RepairResult(
            method_id=unit_for_apply.id,
            status=RepairStatus.VERIFIED,
            original=unit_for_apply.javadoc or "",
            repaired=state["repaired_doc"],
            changes=[
                FieldChange(_field_ref(c["field"]), c["action"], c["before"], c["after"]) for c in state["changes"]
            ],
        )
        new_text, written, skipped = apply_repairs_to_text(content, [(unit_for_apply, result)])
        if not written:
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.issue_number,
                f"{_MARKER}\n⚠️ Could not apply the fix for `{request.method_label}`: "
                f"{skipped[0] if skipped else 'the file changed since the review was posted'}. "
                "Push again to get a fresh review.",
            )
            return
        try:
            commit = self.github.update_file_content(
                event.owner,
                event.repo,
                request.file_path,
                content=new_text,
                sha=sha,
                branch=branch,
                message=f"docdebt: fix documentation for {request.method_label}",
            )
        except GithubApiError as exc:
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.issue_number,
                f"{_MARKER}\n⚠️ GitHub rejected the commit for `{request.method_label}` "
                f"(status {exc.status_code}). The file may have changed since the review was posted; "
                "push again to get a fresh review.",
            )
            return
        commit_sha = commit.get("commit", {}).get("sha", "")[:7]
        self.github.create_issue_comment(
            event.owner,
            event.repo,
            event.issue_number,
            f"{_MARKER}\n✅ Applied the fix for `{request.method_label}` in commit `{commit_sha}`.",
        )


def _field_ref(label: str):
    from docdebt.core.javadoc_edit import DESCRIPTION, FieldRef

    parts = label.split()
    if len(parts) == 1:
        return FieldRef(parts[0].lstrip("@") or DESCRIPTION, None)
    return FieldRef(parts[0].lstrip("@"), parts[1])


def _minimal_unit_for_apply(state: DocDebtState) -> MethodUnit:
    """Rebuild just enough of a :class:`MethodUnit` to call
    :func:`~docdebt.repair.apply.apply_repairs_to_text`, which only reads
    ``file_path``, ``name``, ``start_line``, ``javadoc_start_line`` and
    ``javadoc_end_line`` — never the method's code or signature."""
    from docdebt.core.serialize import unit_from_dict

    return unit_from_dict(state["unit"])
