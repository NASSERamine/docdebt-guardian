"""Minimal GitHub App REST client: just what the review flow needs.

Authentication is the two-step GitHub App dance: sign a short-lived JWT with
the App's private key, exchange it for an installation access token (cached
until shortly before it expires), then use that token like a personal access
token on every REST call. See the GitHub Apps authentication docs for the
exact protocol this follows.

``GithubClient`` takes an ``httpx.Client`` so tests can inject one pointed at
a fake transport instead of the real network.
"""

from __future__ import annotations

import base64
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

API_BASE = "https://api.github.com"
_JWT_TTL_S = 540  # GitHub allows at most 600s; stay under with margin
_TOKEN_REFRESH_MARGIN_S = 60


class GithubApiError(RuntimeError):
    """A GitHub REST call returned an error status."""

    def __init__(self, method: str, url: str, response: httpx.Response) -> None:
        super().__init__(f"{method} {url} -> {response.status_code}: {response.text[:300]}")
        self.status_code = response.status_code
        self.response = response


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # unix timestamp


class GithubClient:
    """Authenticated as one GitHub App installation."""

    def __init__(
        self,
        app_id: str,
        private_key: str,
        installation_id: str,
        *,
        http: httpx.Client | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = app_id
        self.private_key = private_key
        self.installation_id = installation_id
        self.http = http or httpx.Client(timeout=30.0)
        self._now = now
        self._cached: _CachedToken | None = None

    # -- authentication ------------------------------------------------------

    def _app_jwt(self) -> str:
        now = int(self._now())
        payload = {"iat": now - 60, "exp": now + _JWT_TTL_S, "iss": self.app_id}
        return jwt.encode(payload, self.private_key, algorithm="RS256")

    def _installation_token(self) -> str:
        if self._cached and self._cached.expires_at - _TOKEN_REFRESH_MARGIN_S > self._now():
            return self._cached.token
        response = self.http.post(
            f"{API_BASE}/app/installations/{self.installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {self._app_jwt()}", "Accept": "application/vnd.github+json"},
        )
        if response.status_code != 201:
            raise GithubApiError("POST", "installations/access_tokens", response)
        data = response.json()
        expires_at = time.mktime(time.strptime(data["expires_at"], "%Y-%m-%dT%H:%M:%SZ"))
        self._cached = _CachedToken(token=data["token"], expires_at=expires_at)
        return self._cached.token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._installation_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _app_get(self, path: str, **kwargs: Any) -> httpx.Response:
        """A GET authenticated as the App itself (JWT), not as an installation:
        the only way to ask GitHub which installations exist."""
        response = self.http.get(
            f"{API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {self._app_jwt()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            **kwargs,
        )
        if response.status_code >= 400:
            raise GithubApiError("GET", path, response)
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        response = self.http.request(method, f"{API_BASE}{path}", headers=self._headers(), **kwargs)
        if response.status_code >= 400:
            raise GithubApiError(method, path, response)
        return response

    # -- installations (where the App is installed) -----------------------------

    def list_installations(self, *, max_pages: int = 10) -> list[dict]:
        """Every installation of this App (needs no installation id)."""
        found: list[dict] = []
        for page in range(1, max_pages + 1):
            batch = self._app_get("/app/installations", params={"per_page": 100, "page": page}).json()
            found.extend(batch)
            if len(batch) < 100:
                break
        return found

    def list_installation_repositories(self, *, max_pages: int = 10) -> list[dict]:
        """The repositories this client's installation gives access to."""
        found: list[dict] = []
        for page in range(1, max_pages + 1):
            body = self._request(
                "GET", "/installation/repositories", params={"per_page": 100, "page": page}
            ).json()
            found.extend(body.get("repositories", []))
            if len(body.get("repositories", [])) < 100:
                break
        return found

    # -- pull requests --------------------------------------------------------

    def get_pull_request(self, owner: str, repo: str, number: int) -> dict:
        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}").json()

    def list_open_pull_requests(self, owner: str, repo: str, *, max_pages: int = 5) -> list[dict]:
        """The repository's open pull requests, oldest first (100 per page, at
        most ``max_pages`` pages)."""
        pulls: list[dict] = []
        for page in range(1, max_pages + 1):
            batch = self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls",
                params={"state": "open", "per_page": 100, "page": page, "sort": "created", "direction": "asc"},
            ).json()
            pulls.extend(batch)
            if len(batch) < 100:
                break
        return pulls

    def get_pull_request_diff(self, owner: str, repo: str, number: int) -> str:
        response = self.http.request(
            "GET",
            f"{API_BASE}/repos/{owner}/{repo}/pulls/{number}",
            headers={**self._headers(), "Accept": "application/vnd.github.v3.diff"},
        )
        if response.status_code >= 400:
            raise GithubApiError("GET", f"pulls/{number} (diff)", response)
        return response.text

    def get_file_content(self, owner: str, repo: str, path: str, ref: str) -> tuple[str, str]:
        """A file's text content and its blob ``sha`` (needed to update it), at ``ref``."""
        response = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}).json()
        content = base64.b64decode(response["content"]).decode("utf-8")
        return content, response["sha"]

    def update_file_content(
        self, owner: str, repo: str, path: str, *, content: str, sha: str, branch: str, message: str
    ) -> dict:
        """Commit a new version of one file directly to ``branch`` (the PR's head)."""
        body = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "sha": sha,
            "branch": branch,
        }
        return self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", json=body).json()

    # -- comments --------------------------------------------------------------

    def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict:
        """A whole-PR-level comment (PRs are GitHub issues for this endpoint)."""
        return self._request(
            "POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", json={"body": body}
        ).json()

    def add_reaction(self, owner: str, repo: str, comment_id: int, content: str = "eyes") -> None:
        """Acknowledge a slash-command comment (a small courtesy, best-effort:
        never fails the whole flow over a reaction)."""
        with contextlib.suppress(GithubApiError):
            self._request(
                "POST",
                f"/repos/{owner}/{repo}/issues/comments/{comment_id}/reactions",
                json={"content": content},
            )
