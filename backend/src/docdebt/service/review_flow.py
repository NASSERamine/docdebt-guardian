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


def explanation_items(state: DocDebtState) -> list[dict]:
    """What the proposal says about the debt: the exact rules that failed and
    the model's explanation, with the quotes found word for word (highlights)."""
    items = [
        {"source": "rule", "rule": f.get("rule"), "field": f.get("field"), "message": f.get("message"),
         "claim": None, "code": None, "verified": True}
        for f in state.get("structural_findings", [])
        if f.get("severity") == "error"
    ]  # fmt: skip
    for f in state.get("semantic_findings", []):
        items.append(
            {
                "source": "model",
                "rule": f.get("rule"),
                "field": f.get("field"),
                "message": f.get("message"),
                "claim": f.get("quoted_doc"),
                "code": f.get("quoted_code"),
                "verified": bool(f.get("quoted_doc")),
            }
        )
    return items


def proposal_metadata(state: DocDebtState) -> dict:
    """What a proposal is about, stored with it for learning from decisions."""
    import json

    found = [f for f in state.get("structural_findings", []) if f.get("severity") == "error"]
    found += list(state.get("semantic_findings", []))
    rules = sorted({f.get("rule") for f in found if f.get("rule")})
    unit = state.get("unit", {})
    return {
        "rules": ",".join(rules) or None,
        "detection_model": state.get("detection_model"),
        "repair_model": state.get("repair_model"),
        "repair_mechanism": "replacement"
        if state.get("repair_mechanism") == "hybrid"
        else state.get("repair_mechanism"),
        "line": unit.get("javadoc_start_line") or unit.get("start_line"),
        "explanation": json.dumps(explanation_items(state)),
    }


def check_run_fields(pending: list[ReviewRequest], *, block_on_debt: bool, public_url: str | None = None) -> dict:
    """The check run that summarises a PR's pending documentation debt:
    ``success`` when nothing is pending, ``failure`` when something is pending
    and the repository blocks the merge, ``neutral`` otherwise."""
    blocking = block_on_debt and bool(pending)
    if not pending:
        conclusion, title = "success", "No pending documentation debt"
        summary = "No proposal is waiting for a decision on this pull request."
    else:
        conclusion = "failure" if blocking else "neutral"
        title = f"{len(pending)} documentation fix{'es' if len(pending) > 1 else ''} waiting for a decision"
        summary = (
            "Merging is blocked until every proposal is accepted or rejected."
            if blocking
            else "Accept or reject each proposal in its PR comment or in the application."
        )
    lines = []
    for r in pending:
        link = f"{public_url.rstrip('/')}/dashboard/proposals/{r.short_id}" if public_url else None
        where = f"`{r.file_path}:{r.line}`" if r.line else f"`{r.file_path}`"
        lines.append(
            f"- **{r.method_label}** ({where})"
            + (f": {r.rules}" if r.rules else "")
            + (f" — [review]({link})" if link else f" — `/docdebt accept {r.short_id}`")
        )
    annotations = [
        {
            "path": r.file_path,
            "start_line": r.line,
            "end_line": r.line,
            "annotation_level": "failure" if blocking else "warning",
            "title": f"Documentation debt in {r.method_label}",
            "message": f"Documentation debt ({r.rules or 'see proposal'}). "
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


def _diff(before: str, after: str, a: str = "current", b: str = "proposed") -> str:
    from difflib import unified_diff

    return "\n".join(unified_diff(before.splitlines(), after.splitlines(), a, b, n=2, lineterm=""))


def _explanation_markdown(items: list[dict]) -> str:
    lines = []
    for item in items:
        where = f"`{item['field']}`" if item.get("field") else "the documentation"
        text = item.get("message") or ""
        if item.get("source") == "rule":
            lines.append(f"- **Exact rule** `{item.get('rule')}` on {where}: {text}")
        else:
            quote = f' — the documentation says "{item["claim"]}"' if item.get("claim") else ""
            code = f", the code does `{item['code']}`" if item.get("code") else ""
            lines.append(f"- **Model** on {where}: {text}{quote}{code}")
    return "\n".join(lines)


def _proposal_body(
    unit: MethodUnit,
    repaired_doc: str,
    changes: list[dict],
    short_id: str,
    explanation: list[dict] | None = None,
    public_url: str | None = None,
) -> str:
    fields = ", ".join(c["field"] for c in changes) or "the documentation"
    why = _explanation_markdown(explanation or [])
    link = (
        f"\n\n[Review it in the application]({public_url.rstrip('/')}/dashboard/proposals/{short_id})"
        if (public_url)
        else ""
    )
    return (
        f"{_MARKER}\n"
        f"### 📝 Documentation debt in `{unit.file_path}:{unit.start_line}` — "
        f"`{_method_label(unit)}`\n\n"
        + (f"**Why**\n{why}\n\n" if why else "")
        + f"Proposed fix for {fields} (field-by-field replacement):\n\n"
        f"```diff\n{_diff(unit.javadoc or '', repaired_doc)}\n```\n\n"
        f"Reply `/docdebt accept {short_id}` to apply this to the branch, "
        f"`/docdebt regenerate {short_id}` to get a full regeneration to compare, or "
        f"`/docdebt reject {short_id} <reason>` to dismiss it "
        f"(reasons: false-positive, incorrect, manual, other).{link}"
    )


def normalize_edited_javadoc(text: str, indent: str) -> str:
    """A Javadoc typed by a reviewer, laid out like the file: first line
    without indentation (the file's own indentation precedes it), every
    following line indented as the method and starting with ``*``."""
    lines = [line.rstrip() for line in text.strip().replace("\r\n", "\n").split("\n")]
    out = [lines[0].strip()]
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped.startswith("*"):
            stripped = f"* {stripped}" if stripped else "*"
        out.append(f"{indent} {stripped}")
    return "\n".join(out)


def edited_javadoc_problems(unit: MethodUnit, text: str) -> list[str]:
    """Why an edited Javadoc cannot be committed (empty list: it can).

    The text must be one Javadoc block and pass every exact rule of the
    Auditor; an edit identical to the current Javadoc changes nothing."""
    import dataclasses

    from docdebt.core.auditor import Auditor

    stripped = text.strip()
    if not stripped.startswith("/**") or not stripped.endswith("*/") or stripped.count("*/") != 1:
        return ["The text must be a single Javadoc block that starts with /** and ends with */."]
    if " ".join(stripped.split()) == " ".join((unit.javadoc or "").split()):
        return ["The edited text is identical to the current Javadoc: nothing would change."]
    report = Auditor().audit(dataclasses.replace(unit, javadoc=stripped))
    return [f"{f.rule}: {f.message}" for f in report.errors]


class DecisionStatus(StrEnum):
    NOT_FOUND = "not_found"  # unknown id, already resolved, or another PR's id
    REJECTED = "rejected"  # rejected (or the graph did not end in an accepted repair)
    APPLIED = "applied"  # accepted and committed to the PR branch
    NOT_APPLIED = "not_applied"  # accepted, but the commit could not be made
    EXPIRED = "expired"  # the paused graph run behind the proposal no longer exists
    REGENERATED = "regenerated"  # a regeneration was computed and attached to the proposal
    INVALID_EDIT = "invalid_edit"  # the chosen text cannot be committed (edit breaks a rule, no regeneration)


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
    llm_errors: int = 0  # methods whose model call failed (only the exact rules ran)


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
        spec=None,
        model_scope: str | None = None,
        embedder=None,
        progress=None,
    ) -> None:
        self.github = github
        self.graph = graph
        self.store = store
        self.knowledge = knowledge
        self.spec = spec  # the configured model (None: exact rules only)
        self.model_scope = model_scope
        self.embedder = embedder
        self.progress = progress  # callback(done, total) for the scan progress bar
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
        first_error: str | None = None
        for unit in touched:
            thread_id = f"{event.owner}/{event.repo}#{event.number}:{unit.id}"
            cost_before = self.tracker.cost_usd
            state = self.graph.run_unit(unit, thread_id)
            unit_cost = self.tracker.cost_usd - cost_before
            if state.get("detection_error"):
                outcome.llm_errors += 1
                first_error = first_error or state["detection_error"]
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
            self.store.add_proposal_cost(short_id, unit_cost)
            body = _proposal_body(
                unit,
                interrupted.value["repaired_doc"],
                interrupted.value["changes"],
                short_id,
                explanation_items(state),
                self.public_url,
            )
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
        if outcome.llm_errors and first_error:
            self._report_model_failure(event, first_error)
        elif touched and self.spec is None and event.action in ("opened", "reopened"):
            self._report_no_model(event)
        self.publish_check_run(event.owner, event.repo, event.number, event.head_sha)
        return outcome

    # -- controlled degradation: the exact rules ran, the model did not -----------

    def _report_model_failure(self, event: PullRequestEvent, error: str) -> None:
        """The model could not be called: say it on the PR once per review, and
        turn the status banner red (until the next successful connection test)."""
        if self.model_scope:
            self.store.record_llm_failure(self.model_scope, error)
        with contextlib.suppress(GithubApiError):
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.number,
                f"{_MARKER}\n⚠️ DocDebt could not call the model for this push ({error[:300]}). Only the exact "
                "documentation rules ran. Check the API key on the application's Model page, then push again.",
            )

    def _report_no_model(self, event: PullRequestEvent) -> None:
        with contextlib.suppress(GithubApiError):
            self.github.create_issue_comment(
                event.owner,
                event.repo,
                event.number,
                f"{_MARKER}\nℹ️ DocDebt has no model configured for this repository: only the exact "
                "documentation rules ran and no fix can be proposed. Choose a model and its API key on the "
                "application's Model page.",
            )

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
                block_on_debt=policy["block_on_debt"],
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
        prs = self.github.list_open_pull_requests(owner, repo)
        for done, pr in enumerate(prs):
            if self.progress is not None:
                self.progress(done, len(prs))
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
        if command.decision == "regenerate":
            return self.regenerate(
                command.short_id, owner=event.owner, repo=event.repo, pr_number=event.issue_number
            )
        if command.decision == "reject" and command.reason is None:
            with contextlib.suppress(GithubApiError):
                self.github.create_issue_comment(
                    event.owner,
                    event.repo,
                    event.issue_number,
                    f"{_MARKER}\n✋ A rejection needs its reason: `/docdebt reject {command.short_id} "
                    "false-positive` (or `incorrect`, `manual`, `other`).",
                )
            return DecisionOutcome(DecisionStatus.INVALID_EDIT, "A rejection needs a reason.")
        return self.decide(
            command.short_id,
            command.decision,
            owner=event.owner,
            repo=event.repo,
            pr_number=event.issue_number,
            reason=command.reason,
            mechanism=command.mechanism,
        )

    # -- regeneration on demand ------------------------------------------------------

    def _pending_here(self, short_id: str, owner: str, repo: str, pr_number: int) -> ReviewRequest | None:
        request = self.store.get(short_id)
        if (
            request is None
            or request.resolved
            or (request.owner, request.repo, request.pr_number)
            != (
                owner,
                repo,
                pr_number,
            )
        ):
            return None
        return request

    def regenerate(
        self, short_id: str, *, owner: str, repo: str, pr_number: int, via: str | None = None
    ) -> DecisionOutcome:
        """Compute the Regeneration of a pending proposal (one more model call)
        and attach it to the proposal, next to the Replacement. Nothing is
        committed: the reviewer then accepts one of the two."""
        from docdebt.core.serialize import finding_from_dict, unit_from_dict
        from docdebt.rag.retriever import ContextItem, RetrievedContext
        from docdebt.service.engine import build_regenerator

        source = f" (asked from {via})" if via else ""
        request = self._pending_here(short_id, owner, repo, pr_number)
        if request is None:
            return DecisionOutcome(DecisionStatus.NOT_FOUND, f"No pending proposal with id {short_id} here.")
        if self.spec is None:
            return DecisionOutcome(
                DecisionStatus.NOT_APPLIED, "No model is configured: choose one on the Model page first."
            )
        if request.regen_status == "verified" and request.regen_doc:
            return DecisionOutcome(DecisionStatus.REGENERATED, "The regeneration is already available.")
        state = self.graph.get_state(request.thread_id)
        if not state or "unit" not in state:
            return DecisionOutcome(DecisionStatus.EXPIRED, f"The review for {request.method_label} is gone.")
        unit = unit_from_dict(state["unit"])
        findings = [
            finding_from_dict(f) for f in state.get("structural_findings", []) + state.get("semantic_findings", [])
        ]
        context = RetrievedContext(
            [
                ContextItem(i["source"], i["label"], i["path"], i.get("score"), i["text"], i.get("rank", 0))
                for i in state.get("rag_context", [])
            ]
        ).text
        cost_before = self.tracker.cost_usd
        started = time.perf_counter()
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an unscreened wording is shown on the proposal page instead
            result = build_regenerator(self.spec, tracker=self.tracker).repair(
                unit, findings, context=context or None
            )
        self.store.add_proposal_cost(short_id, self.tracker.cost_usd - cost_before)
        self._log_usage("regenerate", owner, repo, pr_number, time.perf_counter() - started)
        if result.status != RepairStatus.VERIFIED or not result.repaired:
            reason = result.reason or result.status.value
            self.store.update_proposal(short_id, regen_status=f"failed: {reason}")
            with contextlib.suppress(GithubApiError):
                self.github.create_issue_comment(
                    owner,
                    repo,
                    pr_number,
                    f"{_MARKER}\n⚠️ No verified regeneration for `{request.method_label}`{source} ({reason}). "
                    f"The field-by-field fix stays available: `/docdebt accept {short_id}`.",
                )
            return DecisionOutcome(DecisionStatus.NOT_APPLIED, f"No verified regeneration ({reason}).")
        self.store.update_proposal(short_id, regen_doc=result.repaired, regen_status="verified")
        with contextlib.suppress(GithubApiError):
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n🔁 Full regeneration for `{request.method_label}`{source}:\n\n"
                f"```diff\n{_diff(unit.javadoc or '', result.repaired, 'current', 'regenerated')}\n```\n\n"
                f"Reply `/docdebt accept {short_id} regenerated` to apply this one instead, or "
                f"`/docdebt accept {short_id}` for the field-by-field fix.",
            )
        return DecisionOutcome(DecisionStatus.REGENERATED, "Regeneration ready: compare it with the replacement.")

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
        mechanism: str = "replacement",
        edited_doc: str | None = None,
    ) -> DecisionOutcome:
        outcome = self._decide(
            short_id,
            decision,
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            via=via,
            reason=reason,
            mechanism=mechanism,
            edited_doc=edited_doc,
        )
        if outcome.status not in (DecisionStatus.NOT_FOUND, DecisionStatus.INVALID_EDIT):
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
        mechanism: str = "replacement",
        edited_doc: str | None = None,
    ) -> DecisionOutcome:
        """Resume the paused graph thread behind ``short_id`` with ``decision``
        ("accept" | "reject") and, on accept, commit the chosen text to the PR
        branch: the replacement (default), the regeneration, or the reviewer's
        edited version (checked by the exact rules first).

        ``owner``/``repo``/``pr_number`` is where the decision was made: a
        command typed on one PR can never resolve another PR's proposal.
        ``via`` names where it came from when that is not the PR itself (e.g.
        "the dashboard"); it is added to the reply comment so the PR keeps a
        trace of every decision. Every outcome is also posted on the PR.
        """
        from docdebt.core.serialize import unit_from_dict

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

        chosen: str | None = None
        if decision == "accept":
            state = self.graph.get_state(request.thread_id)
            unit = unit_from_dict(state["unit"])
            if mechanism == "regeneration":
                if request.regen_status != "verified" or not request.regen_doc:
                    return DecisionOutcome(
                        DecisionStatus.INVALID_EDIT, "No verified regeneration yet: regenerate first."
                    )
                chosen = request.regen_doc
            elif mechanism == "edited":
                chosen = normalize_edited_javadoc(edited_doc or "", unit.indent)
                problems = edited_javadoc_problems(unit, chosen)
                if problems:
                    return DecisionOutcome(DecisionStatus.INVALID_EDIT, " ".join(problems))
            else:
                mechanism, chosen = "replacement", state.get("repaired_doc")

        state = self.graph.resume(request.thread_id, decision)
        self.store.mark_resolved(short_id, decision, reason if decision == "reject" else None)
        self._log_usage("command", owner, repo, pr_number, time.perf_counter() - started)

        if decision != "accept" or state.get("final") != FinalOutcome.ACCEPTED.value or not chosen:
            self.github.create_issue_comment(
                owner,
                repo,
                pr_number,
                f"{_MARKER}\n🚫 Repair for `{request.method_label}` rejected{source} — no changes made.",
            )
            return DecisionOutcome(DecisionStatus.REJECTED, f"Repair for {request.method_label} rejected.")

        self.store.update_proposal(
            short_id, accepted_mechanism=mechanism, edited_doc=chosen if mechanism == "edited" else None
        )
        return self._commit_accepted(request, state, source, chosen, mechanism)

    def _commit_accepted(
        self, request: ReviewRequest, state: DocDebtState, source: str, javadoc: str, mechanism: str
    ) -> DecisionOutcome:
        owner, repo, pr_number = request.owner, request.repo, request.pr_number
        pr = self.github.get_pull_request(owner, repo, pr_number)
        branch = pr["head"]["ref"]
        content, sha = self.github.get_file_content(owner, repo, request.file_path, branch)

        unit_for_apply = _minimal_unit_for_apply(state)
        result = RepairResult(
            method_id=unit_for_apply.id,
            status=RepairStatus.VERIFIED,
            original=unit_for_apply.javadoc or "",
            repaired=javadoc,
            changes=[
                FieldChange(_field_ref(c["field"]), c["action"], c["before"], c["after"])
                for c in state.get("changes", [])
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
        full_sha = commit.get("commit", {}).get("sha", "")
        commit_sha = full_sha[:7]
        self.store.update_proposal(request.short_id, commit_sha=full_sha or None)
        if self.knowledge is not None:
            # An accepted Javadoc (edited ones included) becomes an example of this project's style.
            try:
                self.knowledge.add_validated(owner, repo, unit_for_apply, javadoc, embedder=self.embedder)
            except Exception:
                logger.exception("could not add the accepted Javadoc to the knowledge base")
        label = {
            "replacement": "field-by-field fix",
            "regeneration": "regenerated Javadoc",
            "edited": "edited fix",
        }
        self.github.create_issue_comment(
            owner,
            repo,
            pr_number,
            f"{_MARKER}\n✅ Applied the {label.get(mechanism, 'fix')} for `{request.method_label}` in commit "
            f"`{commit_sha}`{source}.",
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
