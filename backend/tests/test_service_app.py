"""Tests for the FastAPI webhook endpoint: signature verification, event
dispatch, and — the critical safety property — that a comment authored by a
bot (including this service's own posted comments) is never treated as a
slash command.
"""

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from docdebt.service.app import create_app
from docdebt.service.config import ServiceSettings

WEBHOOK_SECRET = "test-secret"


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def app_settings(tmp_path):
    return ServiceSettings(
        github_app_id="1",
        github_app_slug="docdebt-guardian-dev",
        github_private_key="irrelevant-for-these-tests",
        github_webhook_secret=WEBHOOK_SECRET,
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
    )


@pytest.fixture
def client(app_settings, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "docdebt.service.app._handle_pull_request",
        lambda *a, **k: calls.append(("pull_request", a, k)),
    )
    monkeypatch.setattr(
        "docdebt.service.app._handle_comment",
        lambda *a, **k: calls.append(("comment", a, k)),
    )
    app = create_app(app_settings)
    app.state.background_calls = calls  # type: ignore[attr-defined]
    return TestClient(app)


def pr_payload(action="opened"):
    return {
        "action": action,
        "pull_request": {"number": 1, "head": {"sha": "sha1", "ref": "feature"}},
        "repository": {"name": "demo", "owner": {"login": "acme"}},
        "installation": {"id": 555},
    }


def comment_payload(body: str, author_type: str = "User", action: str = "created"):
    return {
        "action": action,
        "issue": {"number": 1, "pull_request": {"url": "..."}},
        "comment": {"id": 42, "body": body, "user": {"type": author_type}},
        "repository": {"name": "demo", "owner": {"login": "acme"}},
        "installation": {"id": 555},
    }


def post_webhook(client, event: str, payload: dict, *, secret: str | None = WEBHOOK_SECRET):
    body = json.dumps(payload).encode()
    headers = {"X-GitHub-Event": event, "X-GitHub-Delivery": "d1", "Content-Type": "application/json"}
    if secret is not None:
        headers["X-Hub-Signature-256"] = sign(body) if secret == WEBHOOK_SECRET else sign(b"wrong-body")
    return client.post("/webhooks/github", content=body, headers=headers)


def test_health():
    from docdebt.service.app import create_app

    settings = ServiceSettings(
        github_app_id="1",
        github_app_slug="docdebt-guardian-dev",
        github_private_key="x",
        github_webhook_secret="s",
        state_db_path="/tmp/does-not-need-to-exist/state.sqlite3",
        graph_checkpoint_path="/tmp/does-not-need-to-exist/graph.sqlite3",
    )
    client = TestClient(create_app(settings))
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_signature_is_rejected(client):
    response = post_webhook(client, "pull_request", pr_payload(), secret=None)
    assert response.status_code == 401


def test_wrong_signature_is_rejected(client):
    body = json.dumps(pr_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": sign(b"not-the-real-body")},
    )
    assert response.status_code == 401


def test_valid_pull_request_opened_is_dispatched(client):
    response = post_webhook(client, "pull_request", pr_payload("opened"))
    assert response.status_code == 202
    calls = client.app.state.background_calls
    assert len(calls) == 1 and calls[0][0] == "pull_request"


def test_pull_request_closed_is_ignored(client):
    response = post_webhook(client, "pull_request", pr_payload("closed"))
    assert response.status_code == 202
    assert client.app.state.background_calls == []


def test_valid_slash_command_from_a_user_is_dispatched(client):
    response = post_webhook(client, "issue_comment", comment_payload("/docdebt accept 1a2b3c4d"))
    assert response.status_code == 202
    calls = client.app.state.background_calls
    assert len(calls) == 1 and calls[0][0] == "comment"


def test_bot_authored_comment_is_never_dispatched_even_with_a_valid_command(client):
    """The critical regression test: the service's OWN proposal comments contain
    the literal text "/docdebt accept <id>" as an instruction for a human to
    type. If a bot-authored comment were ever dispatched, the service would
    see its own proposal echoed back by the webhook and silently accept it —
    with no human involved at all."""
    response = post_webhook(
        client, "issue_comment", comment_payload("/docdebt accept 1a2b3c4d", author_type="Bot")
    )
    assert response.status_code == 202
    assert client.app.state.background_calls == []


def test_comment_with_no_command_is_ignored(client):
    response = post_webhook(client, "issue_comment", comment_payload("just a normal comment, thanks!"))
    assert response.status_code == 202
    assert client.app.state.background_calls == []


def test_unknown_event_type_is_acknowledged_but_ignored(client):
    response = post_webhook(client, "star", {"action": "created"})
    assert response.status_code == 202
    assert client.app.state.background_calls == []


# -- dashboard API and installation webhooks ---------------------------------


def installation_payload(action="created", repos=None, installation_id=555):
    return {
        "action": action,
        "installation": {"id": installation_id},
        "repositories": [{"full_name": name} for name in (repos or ["acme/demo"])],
    }


def installation_repos_payload(action="added", repos=None, installation_id=555):
    key = "repositories_added" if action == "added" else "repositories_removed"
    return {
        "action": action,
        "installation": {"id": installation_id},
        key: [{"full_name": name} for name in (repos or ["acme/demo"])],
    }


def test_installation_created_connects_repositories(client, app_settings):
    from docdebt.service.store import ReviewStore

    response = post_webhook(client, "installation", installation_payload("created"))
    assert response.status_code == 202
    store = ReviewStore(app_settings.state_db_path)
    assert {r["repo"] for r in store.connected_repositories()} == {"demo"}


def test_installation_deleted_disconnects_repositories(client, app_settings):
    from docdebt.service.store import ReviewStore

    post_webhook(client, "installation", installation_payload("created"))
    post_webhook(client, "installation", installation_payload("deleted"))
    store = ReviewStore(app_settings.state_db_path)
    assert store.connected_repositories() == []


def test_installation_repositories_added_and_removed(client, app_settings):
    from docdebt.service.store import ReviewStore

    post_webhook(
        client, "installation_repositories", installation_repos_payload("added", ["acme/one", "acme/two"])
    )
    store = ReviewStore(app_settings.state_db_path)
    assert {r["repo"] for r in store.connected_repositories()} == {"one", "two"}

    post_webhook(client, "installation_repositories", installation_repos_payload("removed", ["acme/one"]))
    assert {r["repo"] for r in store.connected_repositories()} == {"two"}


def test_api_summary_reflects_store_state(client, app_settings):
    from docdebt.service.store import ReviewStore

    store = ReviewStore(app_settings.state_db_path)
    store.create(
        thread_id="t1",
        owner="acme",
        repo="demo",
        pr_number=1,
        file_path="X.java",
        method_label="X.f",
        head_sha="sha1",
    )
    assert client.get("/api/summary").json()["proposals"] == 1


def test_api_repos_merges_connected_and_active(client, app_settings):
    from docdebt.service.store import ReviewStore

    store = ReviewStore(app_settings.state_db_path)
    store.connect_repository(owner="acme", repo="connected-only", installation_id="1")
    store.create(
        thread_id="t1",
        owner="acme",
        repo="active-only",
        pr_number=1,
        file_path="X.java",
        method_label="X.f",
        head_sha="sha1",
    )
    rows = {r["repo"]: r for r in client.get("/api/repos").json()}
    assert rows["connected-only"]["connected"] is True and "proposals" not in rows["connected-only"]
    assert rows["active-only"]["connected"] is False and rows["active-only"]["proposals"] == 1


def test_api_proposals_lists_recent(client, app_settings):
    from docdebt.service.store import ReviewStore

    store = ReviewStore(app_settings.state_db_path)
    store.create(
        thread_id="t1",
        owner="acme",
        repo="demo",
        pr_number=1,
        file_path="X.java",
        method_label="X.f",
        head_sha="sha1",
    )
    proposals = client.get("/api/proposals").json()
    assert len(proposals) == 1 and proposals[0]["method_label"] == "X.f"


def test_api_usage_returns_a_series(client):
    series = client.get("/api/usage?days=3").json()
    assert len(series) == 3 and all("cost_usd" in row for row in series)


def test_api_connect_url_points_at_the_app_install_flow(client):
    body = client.get("/api/connect-url").json()
    assert body["url"] == "https://github.com/apps/docdebt-guardian-dev/installations/new"


# -- dashboard static mount ---------------------------------------------------


def test_dashboard_is_served_when_built(app_settings, tmp_path, monkeypatch):
    from docdebt.service import app as app_module

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="/dashboard/assets/x.js"></script>', encoding="utf-8")
    (dist / "assets" / "x.js").write_text("console.log(1);", encoding="utf-8")
    monkeypatch.setattr(app_module, "_DASHBOARD_DIST", dist)

    client = TestClient(app_module.create_app(app_settings))
    assert client.get("/", follow_redirects=False).status_code == 307
    index = client.get("/dashboard")
    assert index.status_code == 200 and "x.js" in index.text
    assert client.get("/dashboard/some/deep/link").status_code == 200  # SPA fallback
    assert client.get("/dashboard/assets/x.js").status_code == 200


def test_missing_dashboard_build_does_not_break_the_api(app_settings, tmp_path, monkeypatch):
    from docdebt.service import app as app_module

    monkeypatch.setattr(app_module, "_DASHBOARD_DIST", tmp_path / "never-built")
    client = TestClient(app_module.create_app(app_settings))
    assert client.get("/health").status_code == 200
    assert client.get("/api/summary").status_code == 200
    assert client.get("/dashboard").status_code == 404  # no UI mounted, API still works
