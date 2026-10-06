"""FastAPI application: the GitHub App's single webhook endpoint.

GitHub expects a webhook response within about 10 seconds, so the endpoint
only verifies the signature and hands the event to a background task; the
actual graph run (which calls an LLM) happens after the response is sent.
"""

from __future__ import annotations

import dataclasses
import datetime
import hmac
import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docdebt.graph.build import GraphApp, GraphMode, build_graph_deps, sqlite_checkpointer
from docdebt.llm.client import UsageTracker
from docdebt.service.commands import parse_slash_command
from docdebt.service.config import ServiceSettings
from docdebt.service.github_client import GithubApiError, GithubClient
from docdebt.service.proposal_details import build_details, read_thread_values
from docdebt.service.review_flow import DecisionOutcome, DecisionStatus, ReviewFlow
from docdebt.service.runtime_settings import (
    MODEL_CHOICES,
    PATTERN_CHOICES,
    RuntimeSettings,
    default_runtime_settings,
    effective_runtime_settings,
    to_docdebt_settings,
)
from docdebt.service.store import ReviewRequest, ReviewStore
from docdebt.service.webhooks import (
    parse_comment_event,
    parse_installation_event,
    parse_installation_repositories_event,
    parse_pull_request_event,
    verify_signature,
)

logger = logging.getLogger("docdebt.service")

# Serializes every decision (PR comment or dashboard click): resuming the same
# graph thread twice at once could commit the same fix twice.
_DECISION_LOCK = threading.Lock()


class DecisionBody(BaseModel):
    decision: Literal["accept", "reject"]


ProposalStatus = Literal["pending", "accept", "reject", "expired", "unknown"]


def _proposal_json(r: ReviewRequest) -> dict:
    return {
        "short_id": r.short_id,
        "owner": r.owner,
        "repo": r.repo,
        "pr_number": r.pr_number,
        "pr_url": f"https://github.com/{r.owner}/{r.repo}/pull/{r.pr_number}",
        "file_path": r.file_path,
        "method_label": r.method_label,
        "resolved": r.resolved,
        "decision": r.decision,
        "created_at": r.created_at,
    }


def _day_start(day: datetime.date) -> float:
    """Unix timestamp of 00:00 UTC on ``day``."""
    return datetime.datetime.combine(day, datetime.time(), tzinfo=datetime.UTC).timestamp()


def _show_service_logs() -> None:
    """Print this service's INFO logs (reviews, scans, syncs) next to
    uvicorn's: uvicorn only configures its own loggers, so without this they
    are silently dropped. Leaves an already-configured logging setup alone."""
    package_logger = logging.getLogger("docdebt")
    if package_logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:     docdebt - %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    _show_service_logs()
    store = ReviewStore(settings.state_db_path)

    # A scan still "running" in the database was cut short by a restart.
    store.fail_running_scans("interrupted: the service restarted during the scan")
    if settings.sync_installations_on_startup:
        # In a thread: startup must not wait on GitHub, nor fail when it is unreachable.
        threading.Thread(target=_sync_installations_logged, args=(settings, store), daemon=True).start()

    app = FastAPI(title="DocDebt Guardian")
    app.state.settings = settings
    app.state.store = store

    def require_dashboard_token(authorization: str | None = Header(default=None)) -> None:
        """Guard for every dashboard endpoint that changes something: the
        tunnel exposes the whole port, not only the webhook."""
        if settings.dashboard_token is None:
            raise HTTPException(
                status_code=403,
                detail="Changes from the dashboard are disabled: set DOCDEBT_DASHBOARD_TOKEN to enable them.",
            )
        provided = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(provided.encode(), settings.dashboard_token.encode()):
            raise HTTPException(status_code=401, detail="Invalid or missing dashboard token.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/webhooks/github")
    async def github_webhook(
        request: Request,
        background_tasks: BackgroundTasks,
        x_hub_signature_256: str | None = Header(default=None),
        x_github_event: str | None = Header(default=None),
        x_github_delivery: str | None = Header(default=None),
    ) -> Response:
        raw_body = await request.body()
        if not verify_signature(raw_body, x_hub_signature_256, settings.github_webhook_secret):
            raise HTTPException(status_code=401, detail="invalid webhook signature")

        payload = json.loads(raw_body)
        logger.info("webhook received: event=%s delivery=%s", x_github_event, x_github_delivery)

        if x_github_event == "pull_request":
            pr_event = parse_pull_request_event(payload)
            if pr_event is not None:
                background_tasks.add_task(_handle_pull_request, settings, store, pr_event)
        elif x_github_event == "issue_comment":
            comment_event = parse_comment_event(payload)
            if comment_event is not None and not comment_event.from_bot:
                command = parse_slash_command(comment_event.body)
                if command is not None:
                    background_tasks.add_task(_handle_comment, settings, store, comment_event, command)
        elif x_github_event == "installation":
            change = parse_installation_event(payload)
            if change is not None:
                _apply_installation_change(store, change)
        elif x_github_event == "installation_repositories":
            change = parse_installation_repositories_event(payload)
            if change is not None:
                _apply_installation_change(store, change)

        return Response(status_code=202)

    @app.get("/api/summary")
    def api_summary() -> dict:
        return store.summary()

    @app.get("/api/repos")
    def api_repos() -> list[dict]:
        """Every repository the App knows: installed now, installed before
        (``connected`` false, ``installed_before`` true), or with recorded
        activity only."""
        installs = {(r["owner"], r["repo"]): r for r in store.installations()}
        activity = {(r["owner"], r["repo"]): r for r in store.repo_breakdown()}
        rows = []
        for key in sorted(set(installs) | set(activity)):
            owner, repo = key
            install = installs.get(key)
            row = {
                "owner": owner,
                "repo": repo,
                "connected": bool(install and install["active"]),
                "installed_before": bool(install and not install["active"]),
                "connected_at": install["connected_at"] if install else None,
            }
            row.update({k: v for k, v in activity.get(key, {}).items() if k not in ("owner", "repo")})
            rows.append(row)
        return rows

    @app.get("/api/repos/{owner}/{repo}")
    def api_repo_detail(owner: str, repo: str, days: Annotated[int, Query(ge=1, le=365)] = 30) -> dict:
        install = next((r for r in store.installations() if (r["owner"], r["repo"]) == (owner, repo)), None)
        stats = store.repo_stats(owner, repo)
        if install is None and stats["proposals"] == 0 and stats["llm_calls"] == 0:
            raise HTTPException(status_code=404, detail=f"Nothing is known about {owner}/{repo}.")
        return {
            "owner": owner,
            "repo": repo,
            "url": f"https://github.com/{owner}/{repo}",
            "connected": bool(install and install["active"]),
            "installed_before": bool(install and not install["active"]),
            "connected_at": install["connected_at"] if install else None,
            "stats": stats,
            "usage": store.usage_series(days, owner=owner, repo=repo),
            "last_scan": store.latest_scan(owner, repo),
        }

    @app.post("/api/repos/{owner}/{repo}/scan", status_code=202, dependencies=[Depends(require_dashboard_token)])
    def api_scan(owner: str, repo: str, background_tasks: BackgroundTasks) -> dict:
        """Review every open pull request of a connected repository now, as if
        GitHub had sent a webhook for each. Returns at once; poll
        ``GET /api/scans/{id}`` for the result."""
        install = next(
            (r for r in store.installations() if (r["owner"], r["repo"]) == (owner, repo) and r["active"]), None
        )
        if install is None:
            raise HTTPException(
                status_code=409, detail=f"{owner}/{repo} is not connected: install the App on it first."
            )
        cap = effective_runtime_settings(settings, store).max_total_cost_usd
        spent = store.total_cost_usd()
        if cap is not None and spent >= cap:
            raise HTTPException(
                status_code=409,
                detail=f"The LLM cost cap is reached (${spent:.4f} of ${cap:.4f}): raise it in Settings first.",
            )
        scan_id = store.start_scan(owner, repo)
        if scan_id is None:
            raise HTTPException(status_code=409, detail=f"A scan of {owner}/{repo} is already running.")
        background_tasks.add_task(_run_scan, settings, store, scan_id, owner, repo, install["installation_id"])
        return store.get_scan(scan_id)

    @app.post("/api/installations/sync", dependencies=[Depends(require_dashboard_token)])
    def api_sync_installations() -> dict:
        """Ask GitHub which repositories the App is installed on, and update
        the list: for repositories whose installation webhook was missed."""
        try:
            return sync_installations(settings, store)
        except GithubApiError as exc:
            raise HTTPException(status_code=502, detail=f"GitHub refused the request: {exc}") from exc

    @app.get("/api/scans/{scan_id}")
    def api_scan_status(scan_id: int) -> dict:
        scan = store.get_scan(scan_id)
        if scan is None:
            raise HTTPException(status_code=404, detail=f"No scan with id {scan_id}.")
        return scan

    @app.get("/api/proposals")
    def api_proposals(limit: int = 20) -> list[dict]:
        return [_proposal_json(r) for r in store.recent_proposals(limit)]

    @app.get("/api/proposals/search")
    def api_proposals_search(
        repo: Annotated[str | None, Query(description="owner/repo")] = None,
        status: ProposalStatus | None = None,
        q: Annotated[str | None, Query(max_length=200)] = None,
        since: Annotated[datetime.date | None, Query(description="first day included (UTC)")] = None,
        until: Annotated[datetime.date | None, Query(description="last day included (UTC)")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    ) -> dict:
        owner = name = None
        if repo:
            owner, sep, name = repo.partition("/")
            if not sep or not owner or not name:
                raise HTTPException(status_code=422, detail="repo must be written owner/repo.")
        items, total = store.search_proposals(
            owner=owner,
            repo=name,
            status=status,
            text=(q or "").strip() or None,
            since=_day_start(since) if since else None,
            until=_day_start(until + datetime.timedelta(days=1)) if until else None,
            limit=page_size,
            offset=(page - 1) * page_size,
        )
        return {
            "items": [_proposal_json(r) for r in items],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
        }

    @app.get("/api/proposals/{short_id}")
    def api_proposal_detail(short_id: str) -> dict:
        """One proposal with what it is about: the debt found, the method,
        the current and proposed Javadoc and their diff, read back from the
        graph's checkpoint. ``details`` is ``None`` when that run is gone.
        Declared after ``/api/proposals/search`` so "search" is not taken for an id."""
        request = store.get(short_id)
        if request is None:
            raise HTTPException(status_code=404, detail=f"No proposal with id {short_id}.")
        with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:
            values = read_thread_values(saver, request.thread_id)
        details = build_details(values) if values and "unit" in values else None
        return {**_proposal_json(request), "details": details}

    # -- settings ---------------------------------------------------------------

    def _settings_json() -> dict:
        effective = effective_runtime_settings(settings, store)
        return {
            "effective": effective.model_dump(),
            "defaults": default_runtime_settings(settings).model_dump(),
            "overridden": sorted(store.get_settings()),
            "choices": {"models": list(MODEL_CHOICES), "patterns": list(PATTERN_CHOICES)},
            "editable": settings.dashboard_token is not None,
            "spent_usd": round(store.total_cost_usd(), 4),
        }

    @app.get("/api/settings")
    def api_settings() -> dict:
        return _settings_json()

    @app.put("/api/settings", dependencies=[Depends(require_dashboard_token)])
    def api_update_settings(body: RuntimeSettings) -> dict:
        """Save every value of ``body``; they apply from the next webhook on."""
        store.save_settings(body.model_dump())
        return _settings_json()

    @app.delete("/api/settings", dependencies=[Depends(require_dashboard_token)])
    def api_reset_settings() -> dict:
        """Forget every saved value: back to the environment defaults."""
        store.clear_settings()
        return _settings_json()

    @app.get("/api/dashboard-config")
    def api_dashboard_config() -> dict:
        return {"decisions_enabled": settings.dashboard_token is not None}

    @app.post("/api/proposals/{short_id}/decide", dependencies=[Depends(require_dashboard_token)])
    def api_decide(short_id: str, body: DecisionBody) -> dict:
        """Accept or reject a proposal from the dashboard: the same graph resume,
        commit and PR comment as a ``/docdebt accept|reject <id>`` reply.

        Runs synchronously (FastAPI calls a plain ``def`` in its threadpool) so
        the dashboard gets the real outcome, not just "queued".
        """
        request = store.get(short_id)
        if request is None:
            raise HTTPException(status_code=404, detail=f"No proposal with id {short_id}.")
        if request.resolved:
            raise HTTPException(status_code=409, detail=f"Proposal {short_id} is already resolved.")
        installation_id = request.installation_id or store.installation_for(request.owner, request.repo)
        if installation_id is None:
            raise HTTPException(
                status_code=409,
                detail=f"No GitHub installation known for {request.owner}/{request.repo}: "
                f"reply `/docdebt {body.decision} {short_id}` on the pull request instead.",
            )
        try:
            outcome = _decide_from_dashboard(settings, store, request, body.decision, installation_id)
        except Exception as exc:
            logger.exception("dashboard decision failed for %s", short_id)
            raise HTTPException(status_code=502, detail=f"The decision could not be completed: {exc}") from exc
        if outcome.status == DecisionStatus.NOT_FOUND:
            # Resolved by a PR comment between the check above and the lock.
            raise HTTPException(status_code=409, detail=f"Proposal {short_id} is already resolved.")
        return {"status": outcome.status.value, "message": outcome.message, "commit_sha": outcome.commit_sha}

    @app.get("/api/usage")
    def api_usage(days: Annotated[int, Query(ge=1, le=365)] = 14) -> list[dict]:
        return store.usage_series(days)

    @app.get("/api/connect-url")
    def api_connect_url() -> dict:
        """Where the "Connect a repository" button should point: GitHub's own
        installation flow for this App (no token, no manual config on the
        user's side — see docdebt.service README for why)."""
        return {"url": f"https://github.com/apps/{settings.github_app_slug}/installations/new"}

    _mount_dashboard(app)
    return app


_DASHBOARD_DIST = Path(__file__).resolve().parents[3] / "dashboard" / "dist"


def _mount_dashboard(app: FastAPI) -> None:
    """Serve the built dashboard (``npm run build`` in ``dashboard/``) from this
    same process, so the whole product is one process to run: no separate
    frontend server, no CORS configuration, one port to expose.

    Silently does nothing if the dashboard has not been built yet (a fresh
    checkout, or a backend-only test run): the API still works standalone,
    it just has no UI mounted at ``/dashboard``.
    """
    if not (_DASHBOARD_DIST / "index.html").is_file():
        logger.warning(
            "dashboard not built (expected %s); run `npm run build` in dashboard/ to enable /dashboard",
            _DASHBOARD_DIST,
        )
        return
    app.mount("/dashboard/assets", StaticFiles(directory=_DASHBOARD_DIST / "assets"), name="dashboard-assets")

    @app.get("/dashboard")
    @app.get("/dashboard/{_path:path}")
    def dashboard_index(_path: str = "") -> FileResponse:
        # A single-page app: every /dashboard/* route (client-side routing, or
        # just a hard refresh on a deep link) serves the same index.html.
        return FileResponse(_DASHBOARD_DIST / "index.html")

    @app.get("/", include_in_schema=False)
    def root_redirect() -> Response:
        return Response(status_code=307, headers={"Location": "/dashboard"})


def _apply_installation_change(store: ReviewStore, change) -> None:
    for full_name in change.connected:
        owner, _, repo = full_name.partition("/")
        store.connect_repository(owner=owner, repo=repo, installation_id=change.installation_id)
    for full_name in change.disconnected:
        owner, _, repo = full_name.partition("/")
        store.disconnect_repository(owner=owner, repo=repo)


@contextmanager
def _build_flow(settings: ServiceSettings, store: ReviewStore, installation_id: str):
    """One :class:`ReviewFlow`, live only for the duration of the ``with`` block:
    its SQLite checkpointer connection is opened and closed around exactly one
    webhook's worth of work, matching the short-lived-connection pattern used
    everywhere else in this codebase (the checkpointer, ``ReviewStore``).

    The settings are read again here, for every webhook, so a change saved
    from the dashboard applies to the next pull request without a restart."""
    runtime = effective_runtime_settings(settings, store)
    github = GithubClient(settings.github_app_id, settings.github_private_key, installation_id)
    mode: GraphMode = "structural" if runtime.structural_only else "pipeline"
    # One tracker instance shared with every LLMClient the graph will use (Judge and
    # Repairer alike), so ReviewFlow's own usage log reflects real calls, not zeros.
    tracker = UsageTracker()
    deps = build_graph_deps(to_docdebt_settings(runtime), mode=mode, tracker=tracker)
    with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:
        yield ReviewFlow(github, GraphApp(deps, saver), store, tracker=tracker)


def _handle_pull_request(settings, store, event) -> None:
    try:
        cap = effective_runtime_settings(settings, store).max_total_cost_usd
        spent = store.total_cost_usd()
        if cap is not None and spent >= cap:
            logger.warning(
                "cost cap reached ($%.4f of $%.4f): not reviewing %s/%s#%s",
                spent,
                cap,
                event.owner,
                event.repo,
                event.number,
            )
            github = GithubClient(settings.github_app_id, settings.github_private_key, event.installation_id)
            ReviewFlow.post_cost_cap_notice(github, event, spent, cap)
            return
        with _build_flow(settings, store, event.installation_id) as flow:
            flow.review_pull_request(event)
    except Exception:
        logger.exception("failed to review %s/%s#%s", event.owner, event.repo, event.number)


def _handle_comment(settings, store, event, command) -> None:
    try:
        with _DECISION_LOCK, _build_flow(settings, store, event.installation_id) as flow:
            flow.handle_command(event, command)
    except Exception:
        logger.exception("failed to handle command on %s/%s#%s", event.owner, event.repo, event.issue_number)


def sync_installations(settings: ServiceSettings, store: ReviewStore, *, http=None) -> dict:
    """Make the stored installations match GitHub: every repository of every
    installation of the App is connected; a stored one GitHub no longer lists
    is disconnected. Returns the full names connected and disconnected."""
    app_client = GithubClient(settings.github_app_id, settings.github_private_key, "", http=http)
    seen: set[tuple[str, str]] = set()
    for installation in app_client.list_installations():
        installation_id = str(installation["id"])
        client = GithubClient(settings.github_app_id, settings.github_private_key, installation_id, http=http)
        for repository in client.list_installation_repositories():
            owner, repo = repository["owner"]["login"], repository["name"]
            store.connect_repository(owner=owner, repo=repo, installation_id=installation_id)
            seen.add((owner, repo))
    disconnected = []
    for row in store.installations():
        if row["active"] and (row["owner"], row["repo"]) not in seen:
            store.disconnect_repository(owner=row["owner"], repo=row["repo"])
            disconnected.append(f"{row['owner']}/{row['repo']}")
    return {"connected": sorted(f"{o}/{r}" for o, r in seen), "disconnected": sorted(disconnected)}


def _sync_installations_logged(settings: ServiceSettings, store: ReviewStore) -> None:
    try:
        result = sync_installations(settings, store)
        logger.info("installations synced with GitHub: %s", result)
    except Exception:
        logger.exception("could not sync installations with GitHub at startup")


def _run_scan(settings, store, scan_id: int, owner: str, repo: str, installation_id: str) -> None:
    try:
        with _build_flow(settings, store, installation_id) as flow:
            outcome = flow.scan_repository(owner, repo, installation_id)
    except Exception as exc:
        logger.exception("scan %s of %s/%s failed", scan_id, owner, repo)
        store.finish_scan(scan_id, error=f"{type(exc).__name__}: {exc}")
        return
    store.finish_scan(scan_id, **dataclasses.asdict(outcome))
    logger.info("scan %s of %s/%s: %s", scan_id, owner, repo, outcome)


def _decide_from_dashboard(
    settings, store, request: ReviewRequest, decision: str, installation_id: str
) -> DecisionOutcome:
    with _DECISION_LOCK, _build_flow(settings, store, installation_id) as flow:
        return flow.decide(
            request.short_id,
            decision,
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            via="the dashboard",
        )
