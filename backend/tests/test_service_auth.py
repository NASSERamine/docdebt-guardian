"""Sign in with GitHub: the OAuth flow, sessions, what each user may see,
and adding a repository by name. GitHub is a fake transport."""

import urllib.parse

import httpx
import pytest
from fastapi.testclient import TestClient

from docdebt.service.app import create_app
from docdebt.service.auth import parse_repo_ref
from docdebt.service.config import ServiceSettings
from docdebt.service.store import ReviewStore

USER = {"id": 501, "login": "nasser", "name": "Nasser", "avatar_url": "https://avatars/501"}
INSTALLED = {"acme/demo": 10, "acme/newone": 10, "acme/readonly": 10}  # where the App is installed
PERMS = {
    "acme/demo": {"push": True},
    "acme/newone": {"push": True},
    "acme/notinstalled": {"push": True},
    "acme/readonly": {"push": False, "pull": True},
}


def repo_json(full_name):
    owner, name = full_name.split("/")
    return {
        "name": name,
        "full_name": full_name,
        "owner": {"login": owner, "id": 9001},
        "permissions": PERMS[full_name],
    }


def fake_github(request: httpx.Request) -> httpx.Response:
    url, path, auth = request.url, request.url.path, request.headers.get("authorization", "")
    if url.host == "github.com" and path == "/login/oauth/access_token":
        code = urllib.parse.parse_qs(request.content.decode())["code"][0]
        if code == "good":
            return httpx.Response(200, json={"access_token": "user-tok"})
        return httpx.Response(200, json={"error": "bad_verification_code", "error_description": "bad code"})
    if auth == "Bearer app.jwt":  # asked as the App
        name = path.removeprefix("/repos/").removesuffix("/installation")
        if name in INSTALLED:
            return httpx.Response(200, json={"id": INSTALLED[name]})
        return httpx.Response(404, json={"message": "Not Found"})
    assert auth == "Bearer user-tok", f"unexpected credentials for {path}"
    if path == "/user":
        return httpx.Response(200, json=USER)
    if path == "/user/installations":
        return httpx.Response(200, json={"installations": [{"id": 10}]})
    if path == "/user/installations/10/repositories":
        return httpx.Response(200, json={"repositories": [repo_json("acme/demo"), repo_json("acme/readonly")]})
    if path.startswith("/repos/"):
        name = path.removeprefix("/repos/")
        if name in PERMS:
            return httpx.Response(200, json=repo_json(name))
        return httpx.Response(404, json={"message": "Not Found"})
    raise AssertionError(f"unexpected {request.method} {url}")


@pytest.fixture
def settings(tmp_path):
    return ServiceSettings(
        github_app_id="1",
        github_app_slug="docdebt-dev",
        github_private_key="k",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
        github_client_id="Iv1.client",
        github_client_secret="secret",
        public_url="http://localhost:8000",
    )


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "app.jwt")
    http = httpx.Client(transport=httpx.MockTransport(fake_github))
    return TestClient(create_app(settings, github_http=http))


def sign_in(client) -> None:
    login = client.get("/auth/login", follow_redirects=False)
    state = urllib.parse.parse_qs(urllib.parse.urlparse(login.headers["location"]).query)["state"][0]
    callback = client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    assert callback.status_code == 303 and callback.headers["location"] == "/dashboard/overview"


def proposal(settings, owner, repo, label="X.f"):
    return ReviewStore(settings.state_db_path).create(
        thread_id=f"t-{owner}-{repo}-{label}",
        owner=owner,
        repo=repo,
        pr_number=1,
        file_path="X.java",
        method_label=label,
        head_sha="s",
        installation_id="10",
    )


def test_parse_repo_ref():
    assert parse_repo_ref("acme/demo") == ("acme", "demo")
    assert parse_repo_ref("https://github.com/acme/demo/pull/5") == ("acme", "demo")
    assert parse_repo_ref("github.com/acme/demo.git") == ("acme", "demo")
    assert parse_repo_ref("acme") is None and parse_repo_ref("https://gitlab.com/a/b") is None


def test_local_mode_has_no_sign_in(tmp_path):
    local = ServiceSettings(
        "1", "slug", "k", "s", state_db_path=tmp_path / "s.db", graph_checkpoint_path=tmp_path / "g.db"
    )
    client = TestClient(create_app(local))
    assert client.get("/api/me").json() == {"auth_enabled": False, "user": None}
    assert client.get("/api/summary").status_code == 200  # open, as before
    assert client.get("/auth/login", follow_redirects=False).status_code == 404


def test_everything_needs_a_session_in_sign_in_mode(client):
    assert client.get("/api/me").json() == {"auth_enabled": True, "user": None}
    for path in (
        "/api/summary",
        "/api/repos",
        "/api/proposals",
        "/api/proposals/search",
        "/api/usage",
        "/api/settings",
    ):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/repos", json={"repo": "acme/demo"}).status_code == 401
    assert client.get("/health").status_code == 200  # still public


def test_login_redirects_to_github_with_a_state(client):
    response = client.get("/auth/login", follow_redirects=False)
    location = urllib.parse.urlparse(response.headers["location"])
    query = urllib.parse.parse_qs(location.query)
    assert location.netloc == "github.com" and location.path == "/login/oauth/authorize"
    assert query["client_id"] == ["Iv1.client"]
    assert query["redirect_uri"] == ["http://localhost:8000/auth/callback"]
    assert query["state"][0] == response.cookies.get("docdebt_oauth_state")


def test_callback_refuses_a_wrong_state_or_a_bad_code(client):
    client.get("/auth/login", follow_redirects=False)
    wrong = client.get("/auth/callback?code=good&state=forged", follow_redirects=False)
    assert wrong.headers["location"] == "/dashboard/?login_error=state"
    login = client.get("/auth/login", follow_redirects=False)
    state = urllib.parse.parse_qs(urllib.parse.urlparse(login.headers["location"]).query)["state"][0]
    bad = client.get(f"/auth/callback?code=bad&state={state}", follow_redirects=False)
    assert bad.headers["location"] == "/dashboard/?login_error=github"
    assert client.get("/api/me").json()["user"] is None


def test_sign_in_imports_only_the_repositories_the_user_can_push_to(client):
    sign_in(client)
    me = client.get("/api/me").json()["user"]
    assert (me["login"], me["github_id"]) == ("nasser", 501) and "access_token" not in me
    repos = {f"{r['owner']}/{r['repo']}": r for r in client.get("/api/repos").json()}
    assert set(repos) == {"acme/demo"} and repos["acme/demo"]["connected"] is True  # not acme/readonly


def test_a_user_sees_and_acts_only_on_their_repositories(client, settings):
    sign_in(client)
    mine = proposal(settings, "acme", "demo")
    theirs = proposal(settings, "other", "secret")
    assert client.get("/api/summary").json()["proposals"] == 1
    assert [p["short_id"] for p in client.get("/api/proposals").json()] == [mine]
    assert client.get("/api/proposals/search").json()["total"] == 1
    assert client.get("/api/proposals/search?repo=other/secret").json()["total"] == 0
    assert client.get(f"/api/proposals/{theirs}").status_code == 404
    assert client.get("/api/repos/other/secret").status_code == 404
    assert client.post(f"/api/proposals/{theirs}/decide", json={"decision": "accept"}).status_code == 404
    assert client.post("/api/repos/other/secret/scan").status_code == 404
    assert client.get(f"/api/proposals/{mine}").status_code == 200


def test_add_a_repository_by_name_or_url(client):
    sign_in(client)
    added = client.post("/api/repos", json={"repo": "https://github.com/acme/newone"}).json()
    assert added == {"status": "connected", "owner": "acme", "repo": "newone"}
    assert "acme/newone" in {f"{r['owner']}/{r['repo']}" for r in client.get("/api/repos").json()}

    pending = client.post("/api/repos", json={"repo": "acme/notinstalled"}).json()
    assert pending["status"] == "not_installed"
    assert (
        pending["install_url"]
        == "https://github.com/apps/docdebt-dev/installations/new/permissions?target_id=9001"
    )

    assert client.post("/api/repos", json={"repo": "acme/readonly"}).status_code == 403  # read access only
    assert client.post("/api/repos", json={"repo": "acme/missing"}).status_code == 404
    assert client.post("/api/repos", json={"repo": "not a repo"}).status_code == 422


def test_remove_a_repository_and_sign_out(client):
    sign_in(client)
    assert client.delete("/api/repos/acme/demo").status_code == 204
    assert client.get("/api/repos").json() == []
    assert client.post("/auth/logout").status_code == 204
    assert client.get("/api/me").json()["user"] is None
    assert client.get("/api/summary").status_code == 401


def test_decisions_need_no_dashboard_token_when_signed_in(client, settings, monkeypatch):
    from docdebt.service import app as app_module
    from docdebt.service.review_flow import DecisionOutcome, DecisionStatus

    monkeypatch.setattr(
        app_module,
        "_decide_from_dashboard",
        lambda *a, **k: DecisionOutcome(DecisionStatus.REJECTED, "Repair for X.f rejected."),
    )
    sign_in(client)
    mine = proposal(settings, "acme", "demo")
    response = client.post(f"/api/proposals/{mine}/decide", json={"decision": "reject", "reason": "other"})
    assert response.status_code == 200 and response.json()["status"] == "rejected"
    assert client.get("/api/dashboard-config").json() == {"decisions_enabled": True, "auth_enabled": True}
