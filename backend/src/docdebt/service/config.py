"""Service configuration, read from environment variables.

Nothing here is ever committed: every field is a secret or an operational
setting, supplied at process start (a ``.env`` file locally, real secrets in
production). See ``.env.example`` for the full list with explanations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """A required environment variable is missing or unreadable."""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Environment variable '{name}' is required but not set.")
    return value


@dataclass(frozen=True)
class ServiceSettings:
    """Everything the service needs to talk to GitHub and run the graph.

    Attributes:
        github_app_id: the App's numeric id (not the Client ID).
        github_app_slug: the App's URL slug (e.g. "docdebt-guardian-dev"), used
            only to build the "Connect a repository" link the dashboard shows
            (``https://github.com/apps/<slug>/installations/new``).
        github_private_key: the App's PEM private key, as text.
        github_webhook_secret: shared secret used to verify webhook signatures.
        repair_model: model id used for detection and repair (a local Ollama
            model by default, matching the cost-conscious defaults in
            ``.docdebt.yml``).
        repair_pattern: repair prompt pattern.
        structural_only: skip the Judge entirely (deterministic checks only);
            set ``DOCDEBT_STRUCTURAL_ONLY=false`` to enable semantic detection.
        state_db_path: SQLite file mapping the short ids posted in PR comments
            back to graph thread ids (see ``docdebt.service.store``).
        graph_checkpoint_path: SQLite file for the LangGraph checkpointer.
        sync_installations_on_startup: ask GitHub, when the service starts,
            which repositories the App is installed on (repairs a list that
            missed installation webhooks, e.g. an App installed before the
            service tracked them). On by default from the environment.
        github_client_id / github_client_secret: the App's OAuth credentials
            (App page → "Client ID", "Generate a new client secret"). When
            both are set, the dashboard becomes a multi-user application:
            people sign in with GitHub and each one sees only the
            repositories they can push to. When unset: local mode, one
            operator, the dashboard token below.
        public_url: where people open the service (the OAuth callback is
            ``<public_url>/auth/callback``; it must match the App's
            "Callback URL").
        dashboard_token: shared secret the dashboard must send (``Authorization:
            Bearer <token>``) to accept or reject a proposal. ``None`` (the
            default) disables decisions from the dashboard: it stays read-only.
    """

    github_app_id: str
    github_app_slug: str
    github_private_key: str
    github_webhook_secret: str
    repair_model: str = "qwen2.5-coder:7b"
    repair_pattern: str = "zeroshot"
    structural_only: bool = True
    state_db_path: Path = Path(".docdebt-service/state.sqlite3")
    graph_checkpoint_path: Path = Path(".docdebt-service/graph.sqlite3")
    dashboard_token: str | None = None
    github_client_id: str | None = None
    github_client_secret: str | None = None
    public_url: str = "http://localhost:8000"
    sync_installations_on_startup: bool = False

    @property
    def auth_enabled(self) -> bool:
        """Multi-user mode: "Sign in with GitHub" instead of a shared token."""
        return bool(self.github_client_id and self.github_client_secret)

    @classmethod
    def from_env(cls) -> ServiceSettings:
        key_path = os.environ.get("DOCDEBT_GITHUB_PRIVATE_KEY_PATH")
        private_key = (
            Path(key_path).expanduser().read_text(encoding="utf-8")
            if key_path
            else _required("DOCDEBT_GITHUB_PRIVATE_KEY")
        )
        return cls(
            github_app_id=_required("DOCDEBT_GITHUB_APP_ID"),
            github_app_slug=_required("DOCDEBT_GITHUB_APP_SLUG"),
            github_private_key=private_key,
            github_webhook_secret=_required("DOCDEBT_GITHUB_WEBHOOK_SECRET"),
            repair_model=os.environ.get("DOCDEBT_REPAIR_MODEL", "qwen2.5-coder:7b"),
            repair_pattern=os.environ.get("DOCDEBT_REPAIR_PATTERN", "zeroshot"),
            structural_only=os.environ.get("DOCDEBT_STRUCTURAL_ONLY", "true").lower() != "false",
            state_db_path=Path(os.environ.get("DOCDEBT_STATE_DB", ".docdebt-service/state.sqlite3")),
            graph_checkpoint_path=Path(os.environ.get("DOCDEBT_GRAPH_DB", ".docdebt-service/graph.sqlite3")),
            dashboard_token=os.environ.get("DOCDEBT_DASHBOARD_TOKEN") or None,
            github_client_id=os.environ.get("DOCDEBT_GITHUB_CLIENT_ID") or None,
            github_client_secret=os.environ.get("DOCDEBT_GITHUB_CLIENT_SECRET") or None,
            public_url=(os.environ.get("DOCDEBT_PUBLIC_URL") or "http://localhost:8000").rstrip("/"),
            sync_installations_on_startup=os.environ.get("DOCDEBT_SYNC_ON_STARTUP", "true").lower() != "false",
        )
