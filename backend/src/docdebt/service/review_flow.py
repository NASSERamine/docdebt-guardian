"""Orchestration: a pull request event in, PR comments (and, on accept, a
commit) out. Everything Java-specific or LLM-specific is delegated to the
already-validated core (``docdebt.core``, ``docdebt.graph``,
``docdebt.repair``) — this module only adapts them to a GitHub App, which has
no local checkout and must talk to files through the REST API instead of the
filesystem.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from enum import StrEnum

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
CHECK_NAME = "DocDebt Guardian"
_CHECKS_FORBIDDEN: set[str] = set()  # installations without the "Checks" permission (logged once)


def proposal_metadata(state: DocDebtState) -> dict:
    """What a proposal is about, stored with it for learning from decisions."""
    structural = [f for f in state.get("structural_findings", []) if f.get("severity") == "error"]
    semantic = list(state.get("semantic_findings", []))
    kind = "both" if structural and semantic else "structural" if structural else "semantic" if semantic else None
    rules = sorted({f.get("rule") for f in structural + semantic if f.get("rule")})
    unit = state.get("unit", {})
    return {
        "debt_kind": kind,
        "rules": ",".join(rules) or None,
        "confidence": state.get("detection_confidence") if semantic else None,
        "detection_model": state.get("detection_model"),
        "repair_model": state.get("repair_model"),
        "repair_mechanism": state.get("repair_mechanism"),
        "line": unit.get("javadoc_start_line") or unit.get("start_line"),
    }


def check_run_fields(
    pending: list[ReviewRequest], *, block_on_structural: bool, public_url: str | None = None
) -> dict:
    """The check run that summarises a PR's pending documentation debt."""
    structural = [r for r in pending if r.debt_kind in ("structural", "both")]
    blocking = block_on_structural and bool(structural)
    if not pending:
        conclusion, title = "success", "No pending documentation debt"
        summary = "No proposal is waiting for a decision on this pull request."
    else:
        conclusion = "failure" if blocking else "neutral"
        title = f"{len(pending)} documentation fix{'es' if len(pending) > 1 else ''} waiting for a decision"
        summary = f"{len(structural)} structural, {len(pending) - len(structural)} semantic. " + (
            "Merging is blocked until every structural fix is accepted or rejected."
            if blocking
            else "Accept or reject each proposal in its PR comment or in the application."
        )
    lines = []
    for r in pending:
        link = f"{public_url.rstrip('/')}/dashboard/proposals/{r.short_id}" if public_url else None
        where = f"`{r.file_path}:{r.line}`" if r.line else f"`{r.file_path}`"
        lines.append(
            f"- **{r.method_label}** ({where}) — {r.debt_kind or 'debt'}"
            + (f": {r.rules}" if r.rules else "")
            + (f" — [review]({link})" if link else f" — `/docdebt accept {r.short_id}`")
        )
    annotations = [
        {
            "path": r.file_path,
            "start_line": r.line,
            "end_line": r.line,
            "annotation_level": "failure" if blocking and r in structural else "warning",
            "title": f"Documentation debt in {r.method_label}",
            "message": f"{r.debt_kind or 'Documentation'} debt ({r.rules or 'see proposal'}). "
            f"A verified fix is waiting: /docdebt accept {r.short_id}",
        }
        for r in pending
        if r.line
    ][:50]
    output = {"title": title, "summary": summary, "text": "\n".join(lines) or summary}
    if annotations:
        output["annotations"] = annotations
    return {"status": "completed", "conclusion": conclusion, "output": output}


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


class DecisionStatus(StrEnum):
    NOT_FOUND = "not_found"  # unknown id, already resolved, or another PR's id
    REJECTED = "rejected"  # rejected (or the graph did not end in an accepted repair)
    APPLIED = "applied"  # accepted and committed to the PR branch
    NOT_APPLIED = "not_applied"  # accepted, but the commit could not be made
    EXPIRED = "expired"  # the paused graph run behind the proposal no longer exists


@dataclass(frozen=True)
class DecisionOutcome:
    status: DecisionStatus
    message: str
    commit_sha: str | None = None


@dataclass
class ReviewOutcome:
    proposed: int
    skipped_no_debt: int
    skipped_not_proposed: int
    already_proposed: int = 0  # still waiting on a decision: not posted again
    already_decided: int = 0  # accepted or rejected before, for this same content


@dataclass
class ScanOutcome:
    """Totals of an on-demand scan over every open pull request of a repository."""

    prs: int = 0
    proposed: int = 0
    already_proposed: int = 0
    already_decided: int = 0
    clean: int = 0
    not_proposed: int = 0
    failed_prs: int = 0


class ReviewFlow:
    """Bound to one installation's GitHub client, the compiled graph, and the
    short-id store. One instance handles every PR the service is watching."""

    def __init__(
        self,
        github: GithubClient,
        graph: GraphApp,
        store: ReviewStore,
        tracker: UsageTracker | None = None,
        *,
        knowledge=None,
        public_url: str | None = None,
        check_runs: bool = False,
    ) -> None:
        self.github = github
        self.graph = graph
        self.store = store
        self.knowledge = knowledge
        self.public_url = public_url
        self.check_runs = check_runs
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
        rag = getattr(self.graph.deps, "rag", None)
        if rag is not None:
            rag.set_files(file_contents)

        outcome = ReviewOutcome(proposed=0, skipped_no_debt=0, skipped_not_proposed=0, already_proposed=0)
        for unit in touched:
            thread_id = f"{event.owner}/{event.repo}#{event.number}:{unit.id}"
            state = self.graph.run_unit(unit, thread_id)
            interrupted = self.graph.pending_interrupt(thread_id)
            if interrupted is None:
                final = state.get("final")
                if final == FinalOutcome.NO_DEBT.value:
                    outcome.skipped_no_debt += 1
                elif final in (FinalOutcome.ACCEPTED.value, FinalOutcome.REJECTED.value):
                    # A person already decided on a proposal for exactly this
                    # code and Javadoc: the decision stands, nothing is re-proposed.
                    outcome.already_decided += 1
                    logger.info(
                        "%s (%s:%s): already %s for this exact content, not proposed again",
                        _method_label(unit),
                        unit.file_path,
                        unit.start_line,
                        final,
                    )
                else:
                    outcome.skipped_not_proposed += 1
                    logger.info(
                        "no fix proposed for %s (%s:%s): repair %s after %s attempt(s): %s%s",
                        _method_label(unit),
                        unit.file_path,
                        unit.start_line,
                        state.get("repair_status", "not run"),
                        state.get("repair_attempts", 0),
                        state.get("repair_reason") or "no reason recorded",
                        f"; unresolved: {', '.join(state['unresolved'])}" if state.get("unresolved") else "",
                    )
                continue
            if self.store.pending_for_thread(thread_id) is not None:
                # Same paused run, same proposal, already on the PR and waiting
                # for a decision (a push that did not touch this method, or a
                # scan from the dashboard): do not post it again.
                outcome.already_proposed += 1
                continue
            short_id = self.store.create(
                thread_id=thread_id,
                owner=event.owner,
                repo=event.repo,
                pr_number=event.number,
                file_path=unit.file_path,
                method_label=_method_label(unit),
                head_sha=event.head_sha,
                installation_id=event.installation_id,
                metadata=proposal_metadata(state),
            )
            body = _proposal_body(unit, interrupted.value["repaired_doc"], interrupted.value["changes"], short_id)
            self.github.create_issue_comment(event.owner, event.repo, event.number, body)
            outcome.proposed += 1
        logger.info(
            "reviewed %s/%s#%s: %s proposed, %s already proposed, %s already decided, %s clean, %s not proposed",
            event.owner,
            event.repo,
            event.number,
            outcome.proposed,
            outcome.already_proposed,
            outcome.already_decided,
            outcome.skipped_no_debt,
            outcome.skipped_not_proposed,
        )
        self._log_usage("review", event.owner, event.repo, event.number, time.perf_counter() - started)
        self.publish_check_run(event.owner, event.repo, event.number, event.head_sha)
        return outcome

    # -- the PR's check run --------------------------------------------------------

    def publish_check_run(self, owner: str, repo: str, pr_number: int, head_sha: str | None = None) -> None:
        """Create or update the "DocDebt Guardian" check run of the PR's head.
        Best effort: a missing "Checks" permission or an API error never
        breaks a review or a decision."""
        if not self.check_runs or self.github.installation_id in _CHECKS_FORBIDDEN:
            return
        try:
            head_sha = head_sha or self.github.get_pull_request(owner, repo, pr_number)["head"]["sha"]
            policy = self.store.repo_settings(owner, repo)
            fields = check_run_fields(
                self.store.pending_for_pr(owner, repo, pr_number),
                block_on_structural=policy["block_on_structural"],
                public_url=self.public_url,
            )
            self.github.upsert_check_run(owner, repo, head_sha, CHECK_NAME, **fields)
        except GithubApiError as exc:
            if exc.status_code == 403:
                _CHECKS_FORBIDDEN.add(self.github.installation_id)
                logger.warning(
                    'check runs disabled for installation %s: give the GitHub App the "Checks: Read and '
                    'write" permission to publish them',
                    self.github.installation_id,
                )
            else:
                logger.warning("could not publish the check run on %s/%s#%s: %s", owner, repo, pr_number, exc)
        except Exception:
            logger.exception("could not publish the check run on %s/%s#%s", owner, repo, pr_number)

    # -- on-demand scan (dashboard button) ---------------------------------------

    def scan_repository(self, owner: str, repo: str, installation_id: str) -> ScanOutcome:
        """Review every open pull request of ``owner/repo`` now, exactly as if
        GitHub had just sent a webhook for each one. A pull request that fails
        (deleted branch, API error) is counted and skipped, not fatal."""
        scan = ScanOutcome()
        for pr in self.github.list_open_pull_requests(owner, repo):
            scan.prs += 1
            event = PullRequestEvent(
                action="scan",
                owner=owner,
                repo=repo,
                number=pr["number"],
                head_sha=pr["head"]["sha"],
                head_ref=pr["head"]["ref"],
                installation_id=installation_id,
            )
            try:
                outcome = self.review_pull_request(event)
            except Exception:
                logger.exception("scan: failed to review %s/%s#%s", owner, repo, pr["number"])
                scan.failed_prs += 1
                continue
            scan.proposed += outcome.proposed
            scan.already_proposed += outcome.already_proposed
            scan.already_decided += outcome.already_decided
            scan.clean += outcome.skipped_no_debt
            scan.not_proposed += outcome.skipped_not_proposed
        return scan

    @staticmethod
    def post_cost_cap_notice(github: GithubClient, event: PullRequestEvent, spent: float, cap: float) -> None:
        """Tell the PR why it was not reviewed: the LLM cost cap set on the
        dashboard's Settings page is reached. Best effort."""
        with contextlib.suppress(GithubApiError):
            github.create_issue_comment(
                event.owner,
                event.repo,
                event.number,
                f"{_MARKER}\n⏸️ DocDebt did not review this push: the LLM cost cap is reached "
                f"(${spent:.4f} spent, cap ${cap:.4f}). Raise or remove the cap in the dashboard's "
                "Settings, then push again.",
            )

    # -- issue_comment: created (a slash command reply) -----------------------

    def handle_command(self, event: CommentEvent, command: SlashCommand) -> DecisionOutcome:
        self.github.add_reaction(event.owner, event.repo, event.comment_id)
        return self.decide(
            command.short_id,
            command.decision,
            owner=event.owner,
            repo=event.repo,
            pr_number=event.issue_number,
            reason=command.reason,
        )

    # -- a decision, from a PR comment or from the dashboard -------------------

    def decide(
        self,
        short_id: str,
        decision: str,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        via: str | None = None,
        reason: str | None = None,
    ) -> DecisionOutcome:
        outcome = self._decide(
            short_id, decision, owner=owner, repo=repo, pr_number=pr_number, via=via, reason=reason
        )
        if outcome.status != DecisionStatus.NOT_FOUND:
            self.publish_check_run(owner, repo, pr_number)
        return outcome

    def _decide(
        self,
        short_id: str,
        decision: str,
        *,
        owner: str,
        repo: str,
        pr_number: int,
        via: str | None = None,
        reason: str | None = None,
    ) -> DecisionOutcome:
        """Resume the paused graph thread behind ``short_id`` with ``decision``
        ("accept" | "reject") and, on accept, commit the fix to the PR branch.

        ``owner``/``repo``/``pr_number`` is where the decision was made: a
        command typed on one PR can never resolve another PR's proposal.
        ``via`` names where it came from when that is not the PR itself (e.g.
        "the dashboard"); it is added to the reply comment so the PR keeps a
        trace of every decision. Every outcome is also posted on the PR.
        """
        started = time.perf_counter()
        source = f" (decided from {via})" if via else ""
        request = self.store.get(short_id)
        where = (owner, repo, pr_number)
        if request is None or request.resolved or (request.owner, request.repo, request.pr_number) != where:
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n⚠️ No pending review found for id `{short_id}` on this pull request "
                "(it may already be resolved, or belong to a different PR).",
            )
            return DecisionOutcome(DecisionStatus.NOT_FOUND, f"No pending review found for id {short_id}.")

        if self.graph.pending_interrupt(request.thread_id) is None:
            # The graph database lost this run (reset, moved, or the row was
            # created by hand): nothing can be resumed, so the proposal can
            # never be applied. Close it instead of leaving it pending forever.
            self.store.mark_resolved(short_id, "expired")
            with contextlib.suppress(GithubApiError):
                self.github.create_issue_comment(
                    owner,
                    repo,
                    pr_number,
                    f"{_MARKER}\n⚠️ The review for `{request.method_label}` is no longer available{source}. "
                    "Push again to get a fresh review.",
                )
            return DecisionOutcome(
                DecisionStatus.EXPIRED,
                f"The review for {request.method_label} is no longer available: push to the PR again "
                "to get a fresh one.",
            )

        state = self.graph.resume(request.thread_id, decision)
        self.store.mark_resolved(short_id, decision, reason if decision == "reject" else None)
        self._log_usage("command", owner, repo, pr_number, time.perf_counter() - started)

        if decision != "accept" or state.get("final") != FinalOutcome.ACCEPTED.value:
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n🚫 Repair for `{request.method_label}` rejected{source} — no changes made.",
            )
            return DecisionOutcome(DecisionStatus.REJECTED, f"Repair for {request.method_label} rejected.")

        return self._commit_accepted(request, state, source)

    def _commit_accepted(self, request: ReviewRequest, state: DocDebtState, source: str) -> DecisionOutcome:
        owner, repo, pr_number = request.owner, request.repo, request.pr_number
        pr = self.github.get_pull_request(owner, repo, pr_number)
        branch = pr["head"]["ref"]
        content, sha = self.github.get_file_content(owner, repo, request.file_path, branch)

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
            reason = skipped[0] if skipped else "the file changed since the review was posted"
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n⚠️ Could not apply the fix for `{request.method_label}`{source}: "
                f"{reason}. Push again to get a fresh review.",
            )
            return DecisionOutcome(
                DecisionStatus.NOT_APPLIED, f"Could not apply the fix for {request.method_label}: {reason}."
            )
        try:
            commit = self.github.update_file_content(
                owner,
                repo,
                request.file_path,
                content=new_text,
                sha=sha,
                branch=branch,
                message=f"docdebt: fix documentation for {request.method_label}",
            )
        except GithubApiError as exc:
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n⚠️ GitHub rejected the commit for `{request.method_label}`{source} "
                f"(status {exc.status_code}). The file may have changed since the review was posted; "
                "push again to get a fresh review.",
            )
            return DecisionOutcome(
                DecisionStatus.NOT_APPLIED,
                f"GitHub rejected the commit for {request.method_label} (status {exc.status_code}).",
            )
        commit_sha = commit.get("commit", {}).get("sha", "")[:7]
        if self.knowledge is not None:
            # An accepted Javadoc becomes an example of this project's style for later repairs.
            try:
                self.knowledge.add_validated(owner, repo, unit_for_apply, state["repaired_doc"])
            except Exception:
                logger.exception("could not add the accepted Javadoc to the knowledge base")
        self.github.create_issue_comment(
            owner,
            repo,
            pr_number,
            f"{_MARKER}\n✅ Applied the fix for `{request.method_label}` in commit `{commit_sha}`{source}.",
        )
        message = f"Applied the fix for {request.method_label} in commit {commit_sha}."
        return DecisionOutcome(DecisionStatus.APPLIED, message, commit_sha)


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
