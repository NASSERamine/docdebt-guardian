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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docdebt.graph.build import GraphApp, sqlite_checkpointer
from docdebt.llm.catalog import SERVICE_MODELS, service_model_choices
from docdebt.llm.client import UsageTracker
from docdebt.rag.embeddings import choose_embedder, embedder_status
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.vectors import default_index
from docdebt.service import engine
from docdebt.service.auth import (
    SESSION_COOKIE,
    SESSION_TTL_S,
    STATE_COOKIE,
    STATE_TTL_S,
    AuthError,
    GithubUser,
    authorize_url,
    can_push,
    exchange_code,
    install_url,
    new_state,
    parse_repo_ref,
)
from docdebt.service.commands import parse_slash_command
from docdebt.service.config import ServiceSettings
from docdebt.service.engine import build_service_deps, regeneration_screened
from docdebt.service.github_client import GithubApiError, GithubClient
from docdebt.service.health import run_health_scan, run_push_index
from docdebt.service.learning import precision
from docdebt.service.llm_settings import (
    LOCAL_SCOPE,
    build_spec,
    public_config,
    resolve_model,
    scope_for_repo,
    test_connection,
    user_scope,
)
from docdebt.service.proposal_details import build_details, read_thread_values
from docdebt.service.review_flow import DecisionOutcome, DecisionStatus, ReviewFlow, explanation_items
from docdebt.service.runtime_settings import (
    RuntimeSettings,
    default_runtime_settings,
    effective_runtime_settings,
)
from docdebt.service.secrets import decrypt, encrypt, key_hint, load_cipher
from docdebt.service.store import REJECT_REASONS, ReviewRequest, ReviewStore
from docdebt.service.webhooks import (
    parse_comment_event,
    parse_installation_event,
    parse_installation_repositories_event,
    parse_pull_request_event,
    parse_push_event,
    verify_signature,
)

logger = logging.getLogger("docdebt.service")

# Serializes every decision (PR comment or dashboard click): resuming the same
# graph thread twice at once could commit the same fix twice.
_DECISION_LOCK = threading.Lock()


class DecisionBody(BaseModel):
    decision: Literal["accept", "reject"]
    reason: Literal["false_positive", "incorrect_fix", "manual", "other"] | None = None
    # What an accept commits: the field-by-field fix (default), the regeneration, or an edit.
    mechanism: Literal["replacement", "regeneration", "edited"] = "replacement"
    edited_doc: str | None = None


class RepoSettingsBody(BaseModel):
    block_on_debt: bool | None = None


class LlmConfigBody(BaseModel):
    model: str
    api_key: str | None = None  # omitted: keep the saved key (to change only the model or the prices)
    price_in_per_m: float | None = None
    price_out_per_m: float | None = None


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
        "rules": r.rules,
        "detection_model": r.detection_model,
        "repair_model": r.repair_model,
        "repair_mechanism": r.repair_mechanism,
        "reject_reason": r.reject_reason,
        "decided_at": r.decided_at,
        "line": r.line,
        "explanation": r.explanation_items,
        "regen_status": r.regen_status,
        "accepted_mechanism": r.accepted_mechanism,
        "cost_usd": None if r.cost_usd is None else round(r.cost_usd, 6),
        "commit_sha": r.commit_sha,
        "commit_url": f"https://github.com/{r.owner}/{r.repo}/commit/{r.commit_sha}" if r.commit_sha else None,
        "file_url": (f"https://github.com/{r.owner}/{r.repo}/pull/{r.pr_number}/files" if r.file_path else None),
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


@dataclass(frozen=True)
class Viewer:
    """Who is asking. ``visible`` is ``None`` in local mode (one operator sees
    everything), else the "owner/repo" names of the signed-in user's repositories."""

    user: dict | None
    visible: list[str] | None

    def can_see(self, owner: str, repo: str) -> bool:
        return self.visible is None or f"{owner}/{repo}" in self.visible


def create_app(settings: ServiceSettings | None = None, *, github_http: httpx.Client | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()
    _show_service_logs()
    store = ReviewStore(settings.state_db_path)

    # A scan still "running" in the database was cut short by a restart.
    store.fail_running_scans("interrupted: the service restarted during the scan")
    store.fail_running_health_scans("interrupted: the service restarted during the scan")
    if settings.sync_installations_on_startup:
        # In a thread: startup must not wait on GitHub, nor fail when it is unreachable.
        threading.Thread(target=_sync_installations_logged, args=(settings, store), daemon=True).start()

    app = FastAPI(title="DocDebt Guardian")
    app.state.settings = settings
    app.state.store = store
    cipher = load_cipher(Path(settings.state_db_path).parent)
    app.state.cipher = cipher
    knowledge = knowledge_base(settings)

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

    # Calls to GitHub made for the dashboard (sign-in, repository checks);
    # tests inject a fake transport here.
    http = github_http or httpx.Client(timeout=30.0)

    def viewer(request: Request) -> Viewer:
        """Everything in local mode; in sign-in mode, the signed-in user's
        repositories only (401 when nobody is signed in)."""
        if not settings.auth_enabled:
            return Viewer(user=None, visible=None)
        user = store.session_user(request.cookies.get(SESSION_COOKIE))
        if user is None:
            raise HTTPException(status_code=401, detail="Sign in with GitHub first.")
        return Viewer(user=user, visible=store.user_repos(user["github_id"]))

    def change_allowed(request: Request, authorization: str | None = Header(default=None)) -> Viewer:
        """Guard for every endpoint that changes something: a signed-in user in
        sign-in mode, the dashboard token in local mode."""
        if settings.auth_enabled:
            return viewer(request)
        require_dashboard_token(authorization)
        return Viewer(user=None, visible=None)

    def not_found_unless_visible(v: Viewer, owner: str, repo: str) -> None:
        # 404, not 403: do not reveal that a repository someone else works on exists.
        if not v.can_see(owner, repo):
            raise HTTPException(status_code=404, detail=f"Nothing is known about {owner}/{repo}.")

    # -- sign in with GitHub -------------------------------------------------------

    @app.get("/api/me")
    def api_me(request: Request) -> dict:
        user = store.session_user(request.cookies.get(SESSION_COOKIE)) if settings.auth_enabled else None
        return {
            "auth_enabled": settings.auth_enabled,
            "user": {k: user[k] for k in ("github_id", "login", "name", "avatar_url")} if user else None,
        }

    @app.get("/auth/login", include_in_schema=False)
    def auth_login() -> Response:
        if not settings.auth_enabled:
            raise HTTPException(status_code=404, detail="Sign-in is not configured.")
        state = new_state()
        response = Response(status_code=307, headers={"Location": authorize_url(settings, state)})
        response.set_cookie(
            STATE_COOKIE, state, max_age=STATE_TTL_S, httponly=True, samesite="lax", secure=_secure(settings)
        )
        return response

    @app.get("/auth/callback", include_in_schema=False)
    def auth_callback(request: Request, code: str = "", state: str = "") -> Response:
        if not settings.auth_enabled:
            raise HTTPException(status_code=404, detail="Sign-in is not configured.")
        expected = request.cookies.get(STATE_COOKIE) or ""
        if not code or not state or not hmac.compare_digest(state.encode(), expected.encode()):
            return _redirect("/dashboard/?login_error=state")
        try:
            token = exchange_code(settings, code, http)
            profile = GithubUser(token, http).profile()
        except (AuthError, GithubApiError, httpx.HTTPError) as exc:
            logger.warning("sign-in failed: %s", exc)
            return _redirect("/dashboard/?login_error=github")
        store.upsert_user(
            github_id=profile["id"],
            login=profile["login"],
            name=profile.get("name"),
            avatar_url=profile.get("avatar_url"),
        )
        try:
            imported = _import_user_repos(store, GithubUser(token, http), profile["id"])
            logger.info("%s signed in; %s repositories imported", profile["login"], len(imported))
        except (GithubApiError, httpx.HTTPError):
            logger.exception("could not import the repositories of %s", profile["login"])
        session_id = store.create_session(profile["id"], token, SESSION_TTL_S)
        response = _redirect("/dashboard/overview")
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            max_age=SESSION_TTL_S,
            httponly=True,
            samesite="lax",
            secure=_secure(settings),
        )
        response.delete_cookie(STATE_COOKIE)
        return response

    @app.post("/auth/logout")
    def auth_logout(request: Request) -> Response:
        session_id = request.cookies.get(SESSION_COOKIE)
        if session_id:
            store.delete_session(session_id)
        response = Response(status_code=204)
        response.delete_cookie(SESSION_COOKIE)
        return response

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

        try:
            payload = json.loads(raw_body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="the webhook body is not JSON") from exc
        logger.info("webhook received: event=%s delivery=%s", x_github_event, x_github_delivery)
        store.set_meta("last_webhook", x_github_event or "unknown")

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
        elif x_github_event == "push":
            push = parse_push_event(payload)
            if push is not None:
                background_tasks.add_task(_handle_push, settings, store, push)
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
    def api_summary(v: Viewer = Depends(viewer)) -> dict:
        return store.summary(v.visible)

    @app.get("/api/repos")
    def api_repos(v: Viewer = Depends(viewer)) -> list[dict]:
        """Every repository the App knows: installed now, installed before
        (``connected`` false, ``installed_before`` true), or with recorded
        activity only."""
        installs = {(r["owner"], r["repo"]): r for r in store.installations() if v.can_see(r["owner"], r["repo"])}
        activity = {(r["owner"], r["repo"]): r for r in store.repo_breakdown(v.visible)}
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
    def api_repo_detail(
        owner: str, repo: str, days: Annotated[int, Query(ge=1, le=365)] = 30, v: Viewer = Depends(viewer)
    ) -> dict:
        not_found_unless_visible(v, owner, repo)
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

    @app.post("/api/repos")
    def api_add_repo(body: AddRepoBody, v: Viewer = Depends(change_allowed)) -> dict:
        """Add a repository by its exact name or GitHub URL.

        Signed in: it must exist, the user must be able to push to it, and
        the App must be installed on it. Answers ``connected``, or
        ``not_installed`` with the GitHub page that installs the App.
        """
        ref = parse_repo_ref(body.repo)
        if ref is None:
            raise HTTPException(
                status_code=422, detail="Write the repository as owner/name, or paste its GitHub URL."
            )
        owner, repo = ref
        owner_id = None
        if v.user is not None:
            try:
                repository = GithubUser(v.user["access_token"], http).repository(owner, repo)
            except GithubApiError as exc:
                if exc.status_code == 401:
                    raise HTTPException(
                        status_code=401, detail="Your GitHub sign-in expired: sign in again."
                    ) from exc
                raise HTTPException(status_code=502, detail=f"GitHub refused the request: {exc}") from exc
            if repository is None:
                raise HTTPException(
                    status_code=404, detail=f"{owner}/{repo} does not exist, or your GitHub account cannot see it."
                )
            if not can_push(repository):
                raise HTTPException(
                    status_code=403,
                    detail=f"You cannot push to {repository['full_name']}: accepting a fix commits to it, "
                    "so only people with write access can add it.",
                )
            owner, repo = repository["owner"]["login"], repository["name"]  # GitHub's exact spelling
            owner_id = repository["owner"]["id"]
        try:
            installation = GithubClient(
                settings.github_app_id, settings.github_private_key, "", http=http
            ).get_repo_installation(owner, repo)
        except GithubApiError as exc:
            raise HTTPException(status_code=502, detail=f"GitHub refused the request: {exc}") from exc
        if installation is None:
            return {
                "status": "not_installed",
                "owner": owner,
                "repo": repo,
                "install_url": install_url(settings, owner_id),
            }
        store.connect_repository(owner=owner, repo=repo, installation_id=str(installation["id"]))
        if v.user is not None:
            store.add_user_repo(v.user["github_id"], owner, repo)
        return {"status": "connected", "owner": owner, "repo": repo}

    @app.delete("/api/repos/{owner}/{repo}", status_code=204)
    def api_remove_repo(owner: str, repo: str, v: Viewer = Depends(change_allowed)) -> Response:
        """Take a repository off the signed-in user's list (the App stays
        installed: that is managed on GitHub)."""
        if v.user is None:
            raise HTTPException(
                status_code=400, detail="Only in sign-in mode: in local mode, uninstall on GitHub."
            )
        store.remove_user_repo(v.user["github_id"], owner, repo)
        return Response(status_code=204)

    @app.post("/api/repos/{owner}/{repo}/scan", status_code=202)
    def api_scan(
        owner: str, repo: str, background_tasks: BackgroundTasks, v: Viewer = Depends(change_allowed)
    ) -> dict:
        """Review every open pull request of a connected repository now, as if
        GitHub had sent a webhook for each. Returns at once; poll
        ``GET /api/scans/{id}`` for the result."""
        not_found_unless_visible(v, owner, repo)
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

    # -- repository policy, health, knowledge base, learning -----------------------

    def _active_install(owner: str, repo: str) -> dict:
        install = next(
            (r for r in store.installations() if (r["owner"], r["repo"]) == (owner, repo) and r["active"]), None
        )
        if install is None:
            raise HTTPException(
                status_code=409, detail=f"{owner}/{repo} is not connected: install the App on it first."
            )
        return install

    @app.get("/api/repos/{owner}/{repo}/settings")
    def api_repo_settings(owner: str, repo: str, v: Viewer = Depends(viewer)) -> dict:
        not_found_unless_visible(v, owner, repo)
        return store.repo_settings(owner, repo)

    @app.put("/api/repos/{owner}/{repo}/settings")
    def api_update_repo_settings(
        owner: str, repo: str, body: RepoSettingsBody, v: Viewer = Depends(change_allowed)
    ) -> dict:
        not_found_unless_visible(v, owner, repo)
        values = {k: val for k, val in body.model_dump().items() if val is not None}
        return store.save_repo_settings(owner, repo, values)

    @app.post("/api/repos/{owner}/{repo}/health-scan", status_code=202)
    def api_health_scan(
        owner: str, repo: str, background_tasks: BackgroundTasks, v: Viewer = Depends(change_allowed)
    ) -> dict:
        """Measure the documentation health of the default branch and (re)index the
        knowledge base. Exact rules and local embeddings only: no LLM call, no cost.
        Poll ``GET /api/health-scans/{id}`` for its progress."""
        not_found_unless_visible(v, owner, repo)
        install = _active_install(owner, repo)
        snapshot_id = store.start_health_scan(owner, repo)
        if snapshot_id is None:
            raise HTTPException(status_code=409, detail=f"A health scan of {owner}/{repo} is already running.")
        background_tasks.add_task(
            _run_health_scan, settings, store, snapshot_id, owner, repo, install["installation_id"]
        )
        return store.get_health_scan(snapshot_id)

    @app.get("/api/health-scans/{snapshot_id}")
    def api_health_scan_status(snapshot_id: int, v: Viewer = Depends(viewer)) -> dict:
        snap = store.get_health_scan(snapshot_id)
        if snap is None or not v.can_see(snap["owner"], snap["repo"]):
            raise HTTPException(status_code=404, detail=f"No health scan with id {snapshot_id}.")
        return snap

    @app.get("/api/repos/{owner}/{repo}/health")
    def api_repo_health(owner: str, repo: str, v: Viewer = Depends(viewer)) -> dict:
        not_found_unless_visible(v, owner, repo)
        history = store.health_history(owner, repo)
        done = [h for h in history if h["status"] == "done"]
        latest = done[0] if done else None
        previous = done[1] if len(done) > 1 else None
        return {
            "latest": latest,
            "previous_score": previous["score"] if previous else None,
            "running": next((h for h in history if h["status"] == "running"), None),
            "last_error": history[0]["error"] if history and history[0]["status"] == "failed" else None,
            "trend": [
                {"date": h["finished_at"], "score": h["score"], "commit_sha": h["commit_sha"]}
                for h in reversed(done)
            ],
        }

    @app.get("/api/repos/{owner}/{repo}/knowledge")
    def api_repo_knowledge(owner: str, repo: str, v: Viewer = Depends(viewer)) -> dict:
        not_found_unless_visible(v, owner, repo)
        return knowledge.stats(owner, repo)

    @app.get("/api/precision")
    def api_precision(
        repo: Annotated[str | None, Query(description="owner/repo")] = None,
        days: Annotated[int, Query(ge=1, le=3650)] = 30,
        v: Viewer = Depends(viewer),
    ) -> dict:
        owner = name = None
        if repo:
            owner, _, name = repo.partition("/")
            not_found_unless_visible(v, owner, name)
        since = datetime.datetime.now(datetime.UTC).timestamp() - days * 86400
        decided = store.decided_proposals(owner=owner, repo=name, since=since, visible=v.visible)
        return {**precision(decided), "days": days}

    @app.get("/api/costs")
    def api_costs(v: Viewer = Depends(viewer)) -> dict:
        """LLM cost per month, and the total (needs the prices on the Model page)."""
        return {
            "monthly": store.monthly_costs(v.visible),
            "total_usd": round(store.summary(v.visible)["cost_usd"], 4),
            "cap_usd": effective_runtime_settings(settings, store).max_total_cost_usd,
        }

    @app.post("/api/installations/sync")
    def api_sync_installations(v: Viewer = Depends(change_allowed)) -> dict:
        """Ask GitHub which repositories the App is installed on, and update
        the list: for repositories whose installation webhook was missed."""
        try:
            result = sync_installations(settings, store)
            if v.user is not None:
                _import_user_repos(store, GithubUser(v.user["access_token"], http), v.user["github_id"])
            return result
        except GithubApiError as exc:
            raise HTTPException(status_code=502, detail=f"GitHub refused the request: {exc}") from exc

    @app.get("/api/scans/{scan_id}")
    def api_scan_status(scan_id: int, v: Viewer = Depends(viewer)) -> dict:
        scan = store.get_scan(scan_id)
        if scan is None or not v.can_see(scan["owner"], scan["repo"]):
            raise HTTPException(status_code=404, detail=f"No scan with id {scan_id}.")
        return scan

    @app.get("/api/proposals")
    def api_proposals(limit: int = 20, v: Viewer = Depends(viewer)) -> list[dict]:
        return [_proposal_json(r) for r in store.recent_proposals(limit, v.visible)]

    @app.get("/api/proposals/search")
    def api_proposals_search(
        repo: Annotated[str | None, Query(description="owner/repo")] = None,
        status: ProposalStatus | None = None,
        q: Annotated[str | None, Query(max_length=200)] = None,
        since: Annotated[datetime.date | None, Query(description="first day included (UTC)")] = None,
        until: Annotated[datetime.date | None, Query(description="last day included (UTC)")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=100)] = 25,
        v: Viewer = Depends(viewer),
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
            visible=v.visible,
        )
        return {
            "items": [_proposal_json(r) for r in items],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, -(-total // page_size)),
        }

    @app.get("/api/proposals/{short_id}")
    def api_proposal_detail(short_id: str, v: Viewer = Depends(viewer)) -> dict:
        """One proposal with what it is about: the debt found, the method,
        the current and proposed Javadoc and their diff, read back from the
        graph's checkpoint. ``details`` is ``None`` when that run is gone.
        Declared after ``/api/proposals/search`` so "search" is not taken for an id."""
        request = store.get(short_id)
        if request is None or not v.can_see(request.owner, request.repo):
            raise HTTPException(status_code=404, detail=f"No proposal with id {short_id}.")
        with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:
            values = read_thread_values(saver, request.thread_id)
        details = build_details(values) if values and "unit" in values else None
        if details is not None and not request.explanation:
            details["explanation"] = explanation_items(values)
        model = details["repair"]["model"] if details else None
        return {
            **_proposal_json(request),
            "details": details,
            "regeneration": {
                "status": request.regen_status,
                "doc": request.regen_doc,
                "screened": regeneration_screened(model) if model else None,
            },
            "edited_doc": request.edited_doc,
            "pr_cost_usd": store.pr_cost_usd(request.owner, request.repo, request.pr_number),
            "reject_reasons": list(REJECT_REASONS),
        }

    @app.get("/api/review-queue")
    def api_review_queue(v: Viewer = Depends(viewer)) -> list[dict]:
        """Every pending proposal, oldest first: the "To review" mode walks through them."""
        return [_proposal_json(r) for r in store.pending_queue(v.visible)]

    @app.post("/api/proposals/{short_id}/regenerate")
    def api_regenerate(short_id: str, v: Viewer = Depends(change_allowed)) -> dict:
        """Compute the full Regeneration of a pending proposal (one more model call)
        and attach it next to the Replacement; nothing is committed."""
        request = store.get(short_id)
        if request is None or not v.can_see(request.owner, request.repo):
            raise HTTPException(status_code=404, detail=f"No proposal with id {short_id}.")
        if request.resolved:
            raise HTTPException(status_code=409, detail=f"Proposal {short_id} is already resolved.")
        installation_id = request.installation_id or store.installation_for(request.owner, request.repo)
        if installation_id is None:
            raise HTTPException(
                status_code=409, detail=f"No GitHub installation known for {request.owner}/{request.repo}."
            )
        try:
            with (
                _DECISION_LOCK,
                _build_flow(settings, store, installation_id, request.owner, request.repo) as flow,
            ):
                outcome = flow.regenerate(
                    short_id,
                    owner=request.owner,
                    repo=request.repo,
                    pr_number=request.pr_number,
                    via="the application",
                )
        except Exception as exc:
            logger.exception("regeneration failed for %s", short_id)
            raise HTTPException(status_code=502, detail=f"The regeneration could not be completed: {exc}") from exc
        updated = store.get(short_id)
        return {
            "status": outcome.status.value,
            "message": outcome.message,
            "regen_status": updated.regen_status if updated else None,
            "regen_doc": updated.regen_doc if updated else None,
        }

    # -- settings ---------------------------------------------------------------

    def _settings_json() -> dict:
        effective = effective_runtime_settings(settings, store)
        return {
            "effective": effective.model_dump(),
            "defaults": default_runtime_settings(settings).model_dump(),
            "overridden": sorted(k for k in store.get_settings() if k in RuntimeSettings.model_fields),
            "editable": settings.auth_enabled or settings.dashboard_token is not None,
            "spent_usd": round(store.total_cost_usd(), 4),
            "rag_embeddings": settings.rag_embeddings,
        }

    @app.get("/api/settings", dependencies=[Depends(viewer)])
    def api_settings() -> dict:
        return _settings_json()

    @app.put("/api/settings", dependencies=[Depends(change_allowed)])
    def api_update_settings(body: RuntimeSettings) -> dict:
        """Save every value of ``body``; they apply from the next webhook on."""
        store.save_settings(body.model_dump())
        return _settings_json()

    @app.delete("/api/settings", dependencies=[Depends(change_allowed)])
    def api_reset_settings() -> dict:
        """Forget every saved value: back to the environment defaults."""
        store.clear_settings()
        return _settings_json()

    # -- the model (Model page) ----------------------------------------------------

    def _scope(v: Viewer) -> str:
        return user_scope(v.user["github_id"]) if v.user is not None else LOCAL_SCOPE

    def _llm_json(v: Viewer) -> dict:
        return {
            "config": public_config(store.llm_config(_scope(v))),
            "choices": service_model_choices(),
            "editable": settings.auth_enabled or settings.dashboard_token is not None,
        }

    def _spec_from_body(body: LlmConfigBody, v: Viewer):
        if body.model not in SERVICE_MODELS:
            raise HTTPException(
                status_code=422, detail=f"Choose one of the offered models: {list(SERVICE_MODELS)}."
            )
        for price in (body.price_in_per_m, body.price_out_per_m):
            if price is not None and price < 0:
                raise HTTPException(status_code=422, detail="Prices cannot be negative.")
        key = (body.api_key or "").strip()
        if not key:
            saved = store.llm_config(_scope(v))
            if saved is None:
                raise HTTPException(status_code=422, detail="Paste the API key of the provider.")
            key = decrypt(cipher, saved["api_key_enc"])
        spec = build_spec(
            body.model, key, price_in_per_m=body.price_in_per_m, price_out_per_m=body.price_out_per_m
        )
        return spec, key

    @app.get("/api/llm-config")
    def api_llm_config(v: Viewer = Depends(viewer)) -> dict:
        return _llm_json(v)

    @app.post("/api/llm-config/test")
    def api_llm_config_test(body: LlmConfigBody, v: Viewer = Depends(change_allowed)) -> dict:
        """One minimal call with this model and key; nothing is saved."""
        spec, _ = _spec_from_body(body, v)
        ok, error = test_connection(spec, engine.CHAT_FACTORY)
        return {"ok": ok, "error": error}

    @app.put("/api/llm-config")
    def api_save_llm_config(body: LlmConfigBody, v: Viewer = Depends(change_allowed)) -> dict:
        """Test the connection, then save (encrypted key). A failed test saves nothing."""
        spec, key = _spec_from_body(body, v)
        ok, error = test_connection(spec, engine.CHAT_FACTORY)
        if not ok:
            raise HTTPException(status_code=422, detail=f"The provider refused the connection: {error}")
        store.save_llm_config(
            _scope(v),
            model=body.model,
            api_key_enc=encrypt(cipher, key),
            key_hint=key_hint(key),
            price_in_per_m=body.price_in_per_m,
            price_out_per_m=body.price_out_per_m,
        )
        return _llm_json(v)

    @app.delete("/api/llm-config")
    def api_delete_llm_config(v: Viewer = Depends(change_allowed)) -> dict:
        store.delete_llm_config(_scope(v))
        return _llm_json(v)

    # -- status banner and onboarding ------------------------------------------------

    @app.get("/api/status")
    def api_status(request: Request) -> dict:
        """What the status banner and the getting-started guide show. Public, so
        the home page can show it too; without a session it says only that."""
        try:
            v = viewer(request)
        except HTTPException:
            return {"signed_in": False}
        config = store.llm_config(_scope(v))
        visible_installs = [r for r in store.installations() if r["active"] and v.can_see(r["owner"], r["repo"])]
        indexed = [r for r in visible_installs if knowledge.stats(r["owner"], r["repo"])["indexed"]]
        summary = store.summary(v.visible)
        last_webhook = store.get_meta("last_webhook")
        embeddings = embedder_status()
        return {
            "signed_in": True,
            "model": {
                "configured": config is not None,
                "model": config["model"] if config else None,
                "ok": None if config is None or config["last_test_ok"] is None else bool(config["last_test_ok"]),
                "error": config["last_test_error"] if config else None,
            },
            "knowledge": {
                "repositories": len(visible_installs),
                "indexed": len(indexed),
                "not_indexed": sorted(f"{r['owner']}/{r['repo']}" for r in visible_installs if r not in indexed),
                "embeddings": settings.rag_embeddings if embeddings["name"] is None else embeddings["name"],
                "embeddings_error": embeddings["error"],
                "vector_index": knowledge.index.name,
            },
            "webhooks": {
                "last_at": last_webhook["updated_at"] if last_webhook else None,
                "last_event": last_webhook["value"] if last_webhook else None,
            },  # fmt: skip
            "onboarding": [
                {"key": "sign_in", "done": True},
                {"key": "install_app", "done": bool(visible_installs)},
                {"key": "configure_model", "done": bool(config and config["last_test_ok"])},
                {"key": "index_knowledge", "done": bool(indexed)},
                {"key": "first_pull_request", "done": summary["proposals"] > 0 or summary["llm_calls"] > 0},
            ],
        }

    @app.get("/api/dashboard-config")
    def api_dashboard_config() -> dict:
        return {
            "decisions_enabled": settings.auth_enabled or settings.dashboard_token is not None,
            "auth_enabled": settings.auth_enabled,
        }

    @app.post("/api/proposals/{short_id}/decide")
    def api_decide(short_id: str, body: DecisionBody, v: Viewer = Depends(change_allowed)) -> dict:
        """Accept or reject a proposal from the dashboard: the same graph resume,
        commit and PR comment as a ``/docdebt accept|reject <id>`` reply.

        Runs synchronously (FastAPI calls a plain ``def`` in its threadpool) so
        the dashboard gets the real outcome, not just "queued".
        """
        if body.decision == "reject" and body.reason is None:
            raise HTTPException(status_code=422, detail="A rejection needs its reason.")
        if body.decision == "accept" and body.mechanism == "edited" and not (body.edited_doc or "").strip():
            raise HTTPException(status_code=422, detail="Type the edited Javadoc to accept.")
        request = store.get(short_id)
        if request is None or not v.can_see(request.owner, request.repo):
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
            outcome = _decide_from_dashboard(
                settings,
                store,
                request,
                body.decision,
                installation_id,
                reason=body.reason,
                mechanism=body.mechanism,
                edited_doc=body.edited_doc,
            )
        except Exception as exc:
            logger.exception("dashboard decision failed for %s", short_id)
            raise HTTPException(status_code=502, detail=f"The decision could not be completed: {exc}") from exc
        if outcome.status == DecisionStatus.NOT_FOUND:
            # Resolved by a PR comment between the check above and the lock.
            raise HTTPException(status_code=409, detail=f"Proposal {short_id} is already resolved.")
        if outcome.status == DecisionStatus.INVALID_EDIT:
            raise HTTPException(status_code=422, detail=outcome.message)
        return {"status": outcome.status.value, "message": outcome.message, "commit_sha": outcome.commit_sha}

    @app.get("/api/usage")
    def api_usage(days: Annotated[int, Query(ge=1, le=365)] = 14, v: Viewer = Depends(viewer)) -> list[dict]:
        return store.usage_series(days, visible=v.visible)

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


_KNOWLEDGE: dict[str, KnowledgeBase] = {}
_KNOWLEDGE_LOCK = threading.Lock()


def knowledge_base(settings: ServiceSettings) -> KnowledgeBase:
    """The knowledge base of this state database (one Chroma client per process)."""
    key = str(Path(settings.state_db_path).resolve())
    with _KNOWLEDGE_LOCK:
        if key not in _KNOWLEDGE:
            _KNOWLEDGE[key] = KnowledgeBase(settings.state_db_path, index=default_index(settings.state_db_path))
        return _KNOWLEDGE[key]


def _repo_model(settings: ServiceSettings, store: ReviewStore, owner: str | None, repo: str | None):
    """The model configuration that reviews ``owner/repo`` (``None``: none saved)."""
    if not owner or not repo:
        return None
    scope = scope_for_repo(store, auth_enabled=settings.auth_enabled, owner=owner, repo=repo)
    return resolve_model(load_cipher(Path(settings.state_db_path).parent), store, scope)


@contextmanager
def _build_flow(
    settings: ServiceSettings,
    store: ReviewStore,
    installation_id: str,
    owner: str | None = None,
    repo: str | None = None,
    *,
    progress=None,
):
    """One :class:`ReviewFlow`, live only for the duration of the ``with`` block:
    its SQLite checkpointer connection is opened and closed around exactly one
    webhook's worth of work, matching the short-lived-connection pattern used
    everywhere else in this codebase (the checkpointer, ``ReviewStore``).

    The model configuration is read again here, for every webhook, so a change
    saved on the Model page applies to the next pull request without a restart."""
    github = GithubClient(settings.github_app_id, settings.github_private_key, installation_id)
    knowledge = knowledge_base(settings)
    resolved = _repo_model(settings, store, owner, repo)
    embedder = choose_embedder(settings.rag_embeddings)
    # One tracker instance shared with every LLMClient the graph will use (Judge and
    # Repairer alike), so ReviewFlow's own usage log reflects real calls, not zeros.
    tracker = UsageTracker()
    deps = build_service_deps(
        resolved.spec if resolved else None,
        owner=owner,
        repo=repo,
        tracker=tracker,
        knowledge=knowledge,
        embedder=embedder,
    )
    with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:
        yield ReviewFlow(
            github,
            GraphApp(deps, saver),
            store,
            tracker=tracker,
            knowledge=knowledge,
            public_url=settings.public_url,
            check_runs=True,
            spec=resolved.spec if resolved else None,
            model_scope=resolved.scope if resolved else None,
            embedder=embedder,
            progress=progress,
        )


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
        with _build_flow(settings, store, event.installation_id, event.owner, event.repo) as flow:
            flow.review_pull_request(event)
    except Exception:
        logger.exception("failed to review %s/%s#%s", event.owner, event.repo, event.number)


def _handle_comment(settings, store, event, command) -> None:
    try:
        with _DECISION_LOCK, _build_flow(settings, store, event.installation_id, event.owner, event.repo) as flow:
            flow.handle_command(event, command)
    except Exception:
        logger.exception("failed to handle command on %s/%s#%s", event.owner, event.repo, event.issue_number)


class AddRepoBody(BaseModel):
    repo: str


def _secure(settings: ServiceSettings) -> bool:
    return settings.public_url.startswith("https://")


def _redirect(location: str) -> Response:
    return Response(status_code=303, headers={"Location": location})


def _import_user_repos(store: ReviewStore, user: GithubUser, github_id: int) -> list[str]:
    """Add to the user's list every repository where the App is installed
    and they can push; marks those repositories connected too."""
    imported = []
    for repository, installation_id in user.installed_repositories():
        if not can_push(repository):
            continue
        owner, repo = repository["owner"]["login"], repository["name"]
        store.connect_repository(owner=owner, repo=repo, installation_id=installation_id)
        store.add_user_repo(github_id, owner, repo)
        imported.append(f"{owner}/{repo}")
    return imported


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
    def progress(done: int, total: int) -> None:
        store.set_progress("scans", scan_id, done, total)

    try:
        with _build_flow(settings, store, installation_id, owner, repo, progress=progress) as flow:
            outcome = flow.scan_repository(owner, repo, installation_id)
    except Exception as exc:
        logger.exception("scan %s of %s/%s failed", scan_id, owner, repo)
        store.finish_scan(scan_id, error=f"{type(exc).__name__}: {exc}")
        return
    store.finish_scan(scan_id, **dataclasses.asdict(outcome))
    logger.info("scan %s of %s/%s: %s", scan_id, owner, repo, outcome)


def _decide_from_dashboard(
    settings,
    store,
    request: ReviewRequest,
    decision: str,
    installation_id: str,
    reason: str | None = None,
    mechanism: str = "replacement",
    edited_doc: str | None = None,
) -> DecisionOutcome:
    with _DECISION_LOCK, _build_flow(settings, store, installation_id, request.owner, request.repo) as flow:
        return flow.decide(
            request.short_id,
            decision,
            owner=request.owner,
            repo=request.repo,
            pr_number=request.pr_number,
            via="the dashboard",
            reason=reason,
            mechanism=mechanism,
            edited_doc=edited_doc,
        )


def _run_health_scan(settings, store, snapshot_id: int, owner: str, repo: str, installation_id: str) -> None:
    def progress(done: int, total: int) -> None:
        store.set_progress("debt_snapshots", snapshot_id, done, total)

    try:
        github = GithubClient(settings.github_app_id, settings.github_private_key, installation_id)
        values = run_health_scan(
            github,
            owner,
            repo,
            max_files=settings.health_max_files,
            knowledge=knowledge_base(settings),
            embedder=choose_embedder(settings.rag_embeddings),
            progress=progress,
        )
    except Exception as exc:
        logger.exception("health scan %s of %s/%s failed", snapshot_id, owner, repo)
        store.finish_health_scan(snapshot_id, error=f"{type(exc).__name__}: {exc}")
        return
    store.finish_health_scan(snapshot_id, **values)
    logger.info(
        "health scan %s of %s/%s: score %s over %s methods",
        snapshot_id,
        owner,
        repo,
        values["score"],
        values["methods"],
    )


def _handle_push(settings, store, event) -> None:
    """A push to the default branch: re-index only the files it changed."""
    knowledge = knowledge_base(settings)
    if not knowledge.stats(event.owner, event.repo)["indexed"]:
        return  # never indexed: the first full index comes from "Analyse the default branch"
    try:
        github = GithubClient(settings.github_app_id, settings.github_private_key, event.installation_id)
        stats = run_push_index(
            github,
            event.owner,
            event.repo,
            after_sha=event.after_sha,
            changed=event.changed,
            removed=event.removed,
            knowledge=knowledge,
            embedder=choose_embedder(settings.rag_embeddings),
        )
        logger.info("push on %s/%s: knowledge base updated (%s chunks)", event.owner, event.repo, stats["total"])
    except Exception:
        logger.exception("could not update the knowledge base of %s/%s after a push", event.owner, event.repo)
