"""Sign in with GitHub, and what a signed-in user may see and do.

Uses the GitHub App's own user authorization (OAuth web flow): the App page
gives a Client ID and secret, and its "Callback URL" must be
``<public_url>/auth/callback``. A signed-in user gets a random session id in
an HttpOnly cookie; their GitHub user token stays server-side and is only
used to check which repositories they may work on.

A user works on the repositories they can push to: accepting a proposal
commits to that repository, so read access alone is not enough.
"""

from __future__ import annotations

import re
import secrets
import urllib.parse
from dataclasses import dataclass

import httpx

from docdebt.service.config import ServiceSettings
from docdebt.service.github_client import API_BASE, GithubApiError

SESSION_COOKIE = "docdebt_session"
STATE_COOKIE = "docdebt_oauth_state"
SESSION_TTL_S = 7 * 24 * 3600
STATE_TTL_S = 600

_GITHUB = "https://github.com"
_NAME = r"[A-Za-z0-9_.-]+"
_REPO_REF = re.compile(
    rf"^\s*(?:(?:https?://)?(?:www\.)?github\.com/)?(?P<owner>{_NAME})/(?P<repo>{_NAME}?)(?:\.git)?/?(?:[/#?].*)?\s*$"
)


class AuthError(RuntimeError):
    """Signing in failed (bad state, refused code, GitHub unreachable)."""


def new_state() -> str:
    return secrets.token_urlsafe(24)


def authorize_url(settings: ServiceSettings, state: str) -> str:
    query = urllib.parse.urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": f"{settings.public_url}/auth/callback",
            "state": state,
        }
    )
    return f"{_GITHUB}/login/oauth/authorize?{query}"


def exchange_code(settings: ServiceSettings, code: str, http: httpx.Client) -> str:
    """Trade the one-time ``code`` GitHub sent back for a user access token."""
    response = http.post(
        f"{_GITHUB}/login/oauth/access_token",
        data={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code": code,
            "redirect_uri": f"{settings.public_url}/auth/callback",
        },
        headers={"Accept": "application/json"},
    )
    body = response.json() if response.status_code == 200 else {}
    if "access_token" not in body:
        raise AuthError(body.get("error_description") or f"GitHub refused the sign-in ({response.status_code}).")
    return body["access_token"]


def parse_repo_ref(text: str) -> tuple[str, str] | None:
    """``owner/repo`` from "owner/repo" or a GitHub URL; ``None`` if neither."""
    match = _REPO_REF.match(text or "")
    if not match or not match["repo"]:
        return None
    return match["owner"], match["repo"]


@dataclass
class GithubUser:
    """GitHub, as the signed-in user (their user access token)."""

    token: str
    http: httpx.Client

    def _get(self, path: str, **kwargs) -> httpx.Response:
        response = self.http.get(
            f"{API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            **kwargs,
        )
        if response.status_code >= 400:
            raise GithubApiError("GET", path, response)
        return response

    def profile(self) -> dict:
        return self._get("/user").json()

    def repository(self, owner: str, repo: str) -> dict | None:
        """The repository as this user sees it (with their ``permissions``),
        or ``None`` when it does not exist or they cannot see it."""
        try:
            return self._get(f"/repos/{owner}/{repo}").json()
        except GithubApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    def installed_repositories(self, *, max_pages: int = 10) -> list[tuple[dict, str]]:
        """Every repository this user can access through an installation of
        the App, with that installation's id."""
        found: list[tuple[dict, str]] = []
        for installation in self._get("/user/installations", params={"per_page": 100}).json()["installations"]:
            installation_id = str(installation["id"])
            for page in range(1, max_pages + 1):
                body = self._get(
                    f"/user/installations/{installation_id}/repositories", params={"per_page": 100, "page": page}
                ).json()
                found.extend((repository, installation_id) for repository in body.get("repositories", []))
                if len(body.get("repositories", [])) < 100:
                    break
        return found


def can_push(repository: dict) -> bool:
    permissions = repository.get("permissions") or {}
    return bool(permissions.get("push") or permissions.get("admin"))


def install_url(settings: ServiceSettings, owner_id: int | None = None) -> str:
    """GitHub's page to install the App; on a given account when known."""
    base = f"{_GITHUB}/apps/{settings.github_app_slug}/installations/new"
    return f"{base}/permissions?target_id={owner_id}" if owner_id else base
