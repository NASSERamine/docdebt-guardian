"""FastAPI application: the GitHub App's single webhook endpoint.

GitHub expects a webhook response within about 10 seconds, so the endpoint
only verifies the signature and hands the event to a background task; the
actual graph run (which calls an LLM) happens after the response is sent.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from docdebt.graph.build import GraphApp, GraphMode, build_graph_deps, sqlite_checkpointer
from docdebt.llm.client import UsageTracker
from docdebt.service.commands import parse_slash_command
from docdebt.service.config import ServiceSettings
from docdebt.service.github_client import GithubClient
from docdebt.service.review_flow import ReviewFlow
from docdebt.service.store import ReviewStore
from docdebt.service.webhooks import (
    parse_comment_event,
    parse_installation_event,
    parse_installation_repositories_event,
    parse_pull_request_event,
    verify_signature,
)
from docdebt.settings import RepairSettings, RouterSettings, Settings

logger = logging.getLogger("docdebt.service")


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    store = ReviewStore(settings.state_db_path)
    # The router's cascade always names the repair model; whether it is actually
    # consulted for semantic detection is decided by the `mode` passed to
    # build_graph_deps below (structural mode never calls it), not by this list.
    docdebt_settings = Settings(
        router=RouterSettings(cascade=[settings.repair_model]),
        repair=RepairSettings(model=settings.repair_model, pattern=settings.repair_pattern),
    )

    app = FastAPI(title="DocDebt Guardian")
    app.state.settings = settings
    app.state.store = store
    app.state.docdebt_settings = docdebt_settings

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
                background_tasks.add_task(_handle_pull_request, settings, store, docdebt_settings, pr_event)
        elif x_github_event == "issue_comment":
            comment_event = parse_comment_event(payload)
            if comment_event is not None and not comment_event.from_bot:
                command = parse_slash_command(comment_event.body)
                if command is not None:
                    background_tasks.add_task(
                        _handle_comment, settings, store, docdebt_settings, comment_event, command
                    )
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
        connected = {(r["owner"], r["repo"]): r for r in store.connected_repositories()}
        activity = {(r["owner"], r["repo"]): r for r in store.repo_breakdown()}
        rows = []
        for key in sorted(set(connected) | set(activity)):
            owner, repo = key
            row = {"owner": owner, "repo": repo, "connected": key in connected}
            row.update({k: v for k, v in activity.get(key, {}).items() if k not in ("owner", "repo")})
            rows.append(row)
        return rows

    @app.get("/api/proposals")
    def api_proposals(limit: int = 20) -> list[dict]:
        return [
            {
                "short_id": r.short_id,
                "owner": r.owner,
                "repo": r.repo,
                "pr_number": r.pr_number,
                "file_path": r.file_path,
                "method_label": r.method_label,
                "resolved": r.resolved,
                "decision": r.decision,
                "created_at": r.created_at,
            }
            for r in store.recent_proposals(limit)
        ]

    @app.get("/api/usage")
    def api_usage(days: int = 14) -> list[dict]:
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
def _build_flow(settings: ServiceSettings, store: ReviewStore, docdebt_settings: Settings, installation_id: str):
    """One :class:`ReviewFlow`, live only for the duration of the ``with`` block:
    its SQLite checkpointer connection is opened and closed around exactly one
    webhook's worth of work, matching the short-lived-connection pattern used
    everywhere else in this codebase (the checkpointer, ``ReviewStore``)."""
    github = GithubClient(settings.github_app_id, settings.github_private_key, installation_id)
    mode: GraphMode = "structural" if settings.structural_only else "pipeline"
    # One tracker instance shared with every LLMClient the graph will use (Judge and
    # Repairer alike), so ReviewFlow's own usage log reflects real calls, not zeros.
    tracker = UsageTracker()
    deps = build_graph_deps(docdebt_settings, mode=mode, tracker=tracker)
    with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:
        yield ReviewFlow(github, GraphApp(deps, saver), store, tracker=tracker)


def _handle_pull_request(settings, store, docdebt_settings, event) -> None:
    try:
        with _build_flow(settings, store, docdebt_settings, event.installation_id) as flow:
            flow.review_pull_request(event)
    except Exception:
        logger.exception("failed to review %s/%s#%s", event.owner, event.repo, event.number)


def _handle_comment(settings, store, docdebt_settings, event, command) -> None:
    try:
        with _build_flow(settings, store, docdebt_settings, event.installation_id) as flow:
            flow.handle_command(event, command)
    except Exception:
        logger.exception("failed to handle command on %s/%s#%s", event.owner, event.repo, event.issue_number)
