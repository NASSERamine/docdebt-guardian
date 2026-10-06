"""GitHub webhook signature verification and payload parsing.

GitHub signs every webhook body with HMAC-SHA256 using the secret configured
on the App; verifying it is the only thing standing between this service and
anyone on the internet who can guess its URL, so it is checked before the
body is parsed at all.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

_SIGNATURE_PREFIX = "sha256="


def verify_signature(payload: bytes, signature_header: str | None, secret: str) -> bool:
    """True if ``signature_header`` (the ``X-Hub-Signature-256`` header) matches
    ``payload`` under ``secret``. Uses a constant-time comparison."""
    if not signature_header or not signature_header.startswith(_SIGNATURE_PREFIX):
        return False
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    provided = signature_header[len(_SIGNATURE_PREFIX) :]
    return hmac.compare_digest(expected, provided)


@dataclass(frozen=True)
class PullRequestEvent:
    action: str  # "opened", "synchronize", "reopened", ... or "scan" (dashboard button)
    owner: str
    repo: str
    number: int
    head_sha: str
    head_ref: str
    installation_id: str


@dataclass(frozen=True)
class CommentEvent:
    action: str  # "created"
    owner: str
    repo: str
    issue_number: int
    comment_id: int
    body: str
    installation_id: str
    is_pull_request: bool
    author_type: str  # "User" | "Bot" | "Organization" ...

    @property
    def from_bot(self) -> bool:
        """True for a comment authored by any bot account, including this
        service's own — GitHub fires an ``issue_comment`` webhook for every
        comment, including the ones this service just posted through the API,
        so a slash command must never be taken from a bot-authored comment:
        otherwise the service would see its own proposal (which spells out
        "/docdebt accept <id>" as an instruction) and silently accept it,
        with no human ever involved."""
        return self.author_type == "Bot"


def parse_pull_request_event(payload: dict) -> PullRequestEvent | None:
    """Parsed :class:`PullRequestEvent`, or ``None`` for an action this service
    does not act on (only "opened" and "synchronize" trigger a review)."""
    action = payload.get("action")
    if action not in ("opened", "synchronize", "reopened"):
        return None
    pr = payload["pull_request"]
    repo = payload["repository"]
    return PullRequestEvent(
        action=action,
        owner=repo["owner"]["login"],
        repo=repo["name"],
        number=pr["number"],
        head_sha=pr["head"]["sha"],
        head_ref=pr["head"]["ref"],
        installation_id=str(payload["installation"]["id"]),
    )


@dataclass(frozen=True)
class InstallationChangeEvent:
    """A repository was connected or disconnected from the App: covers both
    the ``installation`` event (the whole App installed/removed, action
    "created"/"deleted") and ``installation_repositories`` (individual repos
    added/removed from an existing installation, action "added"/"removed").
    Both are normalized to the same shape since the store only cares about
    which full names ("owner/repo") were connected or disconnected.
    """

    installation_id: str
    connected: list[str]  # ["owner/repo", ...]
    disconnected: list[str]


def _full_names(repos: list[dict]) -> list[str]:
    return [r["full_name"] for r in repos]


def parse_installation_event(payload: dict) -> InstallationChangeEvent | None:
    """Parsed :class:`InstallationChangeEvent` for the ``installation`` event.

    ``created`` connects every repository GitHub lists on the payload
    (``[]`` for an "All repositories" installation, which the dashboard then
    shows as "all repos, none listed individually" — same as the App's own
    ``repository_selection`` field). Any other action (deleted, suspend,
    unsuspend, new_permissions_accepted, ...) either disconnects everything
    the payload lists or is otherwise ignored.
    """
    installation_id = str(payload["installation"]["id"])
    action = payload.get("action")
    repos = _full_names(payload.get("repositories", []))
    if action == "created":
        return InstallationChangeEvent(installation_id, connected=repos, disconnected=[])
    if action == "deleted":
        return InstallationChangeEvent(installation_id, connected=[], disconnected=repos)
    return None


def parse_installation_repositories_event(payload: dict) -> InstallationChangeEvent | None:
    """Parsed :class:`InstallationChangeEvent` for the
    ``installation_repositories`` event (repos added/removed from an
    installation that already exists, e.g. via "Configure" on GitHub)."""
    if payload.get("action") != "added" and payload.get("action") != "removed":
        return None
    installation_id = str(payload["installation"]["id"])
    return InstallationChangeEvent(
        installation_id,
        connected=_full_names(payload.get("repositories_added", [])),
        disconnected=_full_names(payload.get("repositories_removed", [])),
    )


def parse_comment_event(payload: dict) -> CommentEvent | None:
    """Parsed :class:`CommentEvent`, or ``None`` for an action this service does
    not act on (only a newly created comment can trigger a resume)."""
    if payload.get("action") != "created":
        return None
    issue = payload["issue"]
    repo = payload["repository"]
    comment = payload["comment"]
    return CommentEvent(
        action="created",
        owner=repo["owner"]["login"],
        repo=repo["name"],
        issue_number=issue["number"],
        comment_id=comment["id"],
        body=comment["body"],
        installation_id=str(payload["installation"]["id"]),
        is_pull_request="pull_request" in issue,
        author_type=comment.get("user", {}).get("type", "User"),
    )
