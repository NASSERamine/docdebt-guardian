"""Backend of the dashboard's pages: history search, repository detail,
settings (stored overrides applied to every review), and the cost cap."""

import time

import pytest
from fastapi.testclient import TestClient

from docdebt.service.app import create_app
from docdebt.service.config import ServiceSettings
from docdebt.service.runtime_settings import effective_runtime_settings, to_docdebt_settings
from docdebt.service.store import ReviewStore

TOKEN = "dash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def settings(tmp_path):
    return ServiceSettings(
        github_app_id="1",
        github_app_slug="docdebt-guardian-dev",
        github_private_key="irrelevant",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
        dashboard_token=TOKEN,
    )


@pytest.fixture
def store(settings):
    return ReviewStore(settings.state_db_path)


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def add(store, *, repo="demo", owner="acme", pr=1, label="X.f", path="X.java", created_at=None):
    short_id = store.create(
        thread_id=f"t-{label}-{pr}",
        owner=owner,
        repo=repo,
        pr_number=pr,
        file_path=path,
        method_label=label,
        head_sha="sha",
        installation_id="1",
    )
    if created_at is not None:
        with store._connect() as conn:
            conn.execute("UPDATE review_requests SET created_at = ? WHERE short_id = ?", (created_at, short_id))
    return short_id


# -- store ----------------------------------------------------------------------


def test_search_filters_by_status_repo_and_text(store):
    a = add(store, label="Temperature.toCelsius")
    b = add(store, label="Cache.get", repo="other")
    c = add(store, label="Cache_put", path="src/Cache.java")
    store.mark_resolved(b, "accept")
    store.mark_resolved(c, "expired")

    def ids(**kw):
        return {r.short_id for r in store.search_proposals(**kw)[0]}

    assert ids(status="pending") == {a}
    assert ids(status="accept") == {b}
    assert ids(status="expired") == {c}
    assert ids(owner="acme", repo="other") == {b}
    assert ids(text="cache") == {b, c}  # case-insensitive, method label or path
    assert ids(text="Cache_") == {c}  # "_" is literal, not a LIKE wildcard
    assert ids(text="%") == set()  # "%" is literal too
    assert ids(text=a[:6]) == {a}  # short id
    with pytest.raises(ValueError):
        store.search_proposals(status="merged")


def test_search_paginates_newest_first_with_a_total(store):
    now = time.time()
    created = [add(store, label=f"M.m{i}", created_at=now - i) for i in range(7)]
    page1, total = store.search_proposals(limit=3, offset=0)
    page3, _ = store.search_proposals(limit=3, offset=6)
    assert total == 7
    assert [r.short_id for r in page1] == created[:3]
    assert [r.short_id for r in page3] == created[6:]


def test_search_by_date_range(store):
    day = 86400
    old = add(store, label="Old.m", created_at=1_000 * day)
    new = add(store, label="New.m", created_at=1_010 * day)
    assert {r.short_id for r in store.search_proposals(since=1_005 * day)[0]} == {new}
    assert {r.short_id for r in store.search_proposals(until=1_005 * day)[0]} == {old}


def test_repo_stats_and_per_repo_usage(store):
    add(store, repo="demo")
    store.mark_resolved(add(store, repo="demo", label="Y.g"), "reject")
    add(store, repo="other")
    for repo, calls in (("demo", 3), ("other", 5)):
        store.log_usage(
            owner="acme",
            repo=repo,
            pr_number=1,
            kind="review",
            calls=calls,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.5,
            latency_s=1.0,
        )
    stats = store.repo_stats("acme", "demo")
    assert (stats["proposals"], stats["pending"], stats["rejected"], stats["llm_calls"]) == (2, 1, 1, 3)
    assert stats["cost_usd"] == 0.5 and stats["tokens"] == 15
    assert store.repo_stats("acme", "nothing")["proposals"] == 0
    assert store.usage_series(1, owner="acme", repo="demo")[-1]["calls"] == 3
    assert store.usage_series(1)[-1]["calls"] == 8
    assert store.total_cost_usd() == 1.0


def test_installations_keep_disconnected_repositories(store):
    store.connect_repository(owner="acme", repo="a", installation_id="1")
    store.connect_repository(owner="acme", repo="b", installation_id="1")
    store.disconnect_repository(owner="acme", repo="b")
    assert {(r["repo"], r["active"]) for r in store.installations()} == {("a", True), ("b", False)}


def test_settings_overrides_round_trip(store):
    assert store.get_settings() == {}
    store.save_settings({"structural_only": False, "max_total_cost_usd": 2.5})
    store.save_settings({"max_total_cost_usd": None})
    assert store.get_settings() == {"structural_only": False, "max_total_cost_usd": None}
    store.clear_settings()
    assert store.get_settings() == {}


# -- runtime settings -------------------------------------------------------------


def test_effective_settings_are_env_defaults_plus_saved_overrides(settings, store):
    assert effective_runtime_settings(settings, store).structural_only is True
    store.save_settings({"structural_only": False, "confidence_threshold": 0.8, "repair_model": "codegemma:7b"})
    runtime = effective_runtime_settings(settings, store)
    assert (runtime.structural_only, runtime.confidence_threshold, runtime.repair_model) == (
        False,
        0.8,
        "codegemma:7b",
    )
    docdebt = to_docdebt_settings(runtime)
    assert docdebt.repair.model == "codegemma:7b" and docdebt.router.cascade == ["codegemma:7b"]
    assert docdebt.judge_config("codegemma:7b").confidence_threshold == 0.8


def test_an_invalid_saved_value_is_ignored_not_fatal(settings, store):
    store.save_settings({"repair_model": "no-such-model", "confidence_threshold": 7, "structural_only": False})
    runtime = effective_runtime_settings(settings, store)
    assert runtime.repair_model == settings.repair_model  # bad values fall back to the defaults
    assert runtime.confidence_threshold == 0.0
    assert runtime.structural_only is False  # the valid one still applies


# -- API: history and repository detail -------------------------------------------


def test_api_search_returns_a_page_and_its_count(client, store):
    for i in range(30):
        add(store, label=f"M.m{i}")
    body = client.get("/api/proposals/search?page=2&page_size=25").json()
    assert (body["total"], body["page"], body["pages"], len(body["items"])) == (30, 2, 2, 5)
    item = body["items"][0]
    assert item["pr_url"] == "https://github.com/acme/demo/pull/1"


def test_api_search_filters_and_validation(client, store):
    add(store, repo="demo", label="A.a")
    add(store, repo="other", label="B.b")
    assert client.get("/api/proposals/search?repo=acme/other").json()["total"] == 1
    assert client.get("/api/proposals/search?status=pending&q=A.a").json()["total"] == 1
    today = time.strftime("%Y-%m-%d", time.gmtime())
    assert client.get(f"/api/proposals/search?since={today}&until={today}").json()["total"] == 2
    assert client.get("/api/proposals/search?since=2000-01-01&until=2000-01-02").json()["total"] == 0
    assert client.get("/api/proposals/search?repo=nodash").status_code == 422
    assert client.get("/api/proposals/search?status=merged").status_code == 422
    assert client.get("/api/proposals/search?page_size=1000").status_code == 422


def test_api_repo_detail(client, store):
    add(store, repo="demo")
    store.connect_repository(owner="acme", repo="quiet", installation_id="1")
    detail = client.get("/api/repos/acme/demo?days=7").json()
    assert detail["stats"]["proposals"] == 1 and len(detail["usage"]) == 7
    assert detail["connected"] is False and detail["url"] == "https://github.com/acme/demo"
    assert client.get("/api/repos/acme/quiet").json()["connected"] is True  # installed, no activity yet
    assert client.get("/api/repos/acme/unknown").status_code == 404


def test_api_repos_lists_disconnected_installations(client, store):
    store.connect_repository(owner="acme", repo="gone", installation_id="1")
    store.disconnect_repository(owner="acme", repo="gone")
    rows = {r["repo"]: r for r in client.get("/api/repos").json()}
    assert rows["gone"]["connected"] is False and rows["gone"]["installed_before"] is True


# -- API: settings ---------------------------------------------------------------------


def valid_settings(**overrides):
    return {
        "repair_model": "qwen2.5-coder:7b",
        "repair_pattern": "cot",
        "structural_only": False,
        "confidence_threshold": 0.6,
        "max_total_cost_usd": 1.5,
        **overrides,
    }


def test_api_settings_read_update_and_reset(client):
    initial = client.get("/api/settings").json()
    assert initial["editable"] is True and initial["overridden"] == []
    assert "qwen2.5-coder:7b" in initial["choices"]["models"]

    updated = client.put("/api/settings", json=valid_settings(), headers=AUTH).json()
    assert updated["effective"]["repair_pattern"] == "cot" and updated["effective"]["max_total_cost_usd"] == 1.5
    assert "max_total_cost_usd" in updated["overridden"]
    assert client.get("/api/settings").json()["effective"]["confidence_threshold"] == 0.6

    reset = client.delete("/api/settings", headers=AUTH).json()
    assert reset["effective"] == initial["effective"] and reset["overridden"] == []


def test_api_settings_changes_need_the_token_and_valid_values(client, settings):
    assert client.put("/api/settings", json=valid_settings()).status_code == 401
    assert client.delete("/api/settings").status_code == 401
    assert client.put("/api/settings", json=valid_settings(repair_model="gpt-9"), headers=AUTH).status_code == 422
    assert (
        client.put("/api/settings", json=valid_settings(repair_pattern="oneshot"), headers=AUTH).status_code == 422
    )
    assert (
        client.put("/api/settings", json=valid_settings(confidence_threshold=2), headers=AUTH).status_code == 422
    )
    assert client.put("/api/settings", json=valid_settings(max_total_cost_usd=-1), headers=AUTH).status_code == 422

    from dataclasses import replace

    read_only = TestClient(create_app(replace(settings, dashboard_token=None)))
    assert read_only.get("/api/settings").json()["editable"] is False
    assert read_only.put("/api/settings", json=valid_settings()).status_code == 403


# -- settings reach the review run; the cost cap stops it ---------------------------


def pr_event():
    from docdebt.service.webhooks import PullRequestEvent

    return PullRequestEvent(
        action="opened", owner="acme", repo="demo", number=1, head_sha="s", head_ref="f", installation_id="1"
    )


def test_a_review_uses_the_settings_saved_from_the_dashboard(settings, store, monkeypatch):
    from docdebt.service import app as app_module

    seen = {}

    def fake_deps(docdebt_settings, *, mode, tracker):
        seen.update(model=docdebt_settings.repair.model, pattern=docdebt_settings.repair.pattern, mode=mode)
        raise RuntimeError("stop after capturing the configuration")

    monkeypatch.setattr(app_module, "build_graph_deps", fake_deps)
    store.save_settings({"repair_model": "deepseek-coder:6.7b-instruct", "repair_pattern": "cot"})
    store.save_settings({"structural_only": False})
    app_module._handle_pull_request(settings, store, pr_event())  # the error is logged, not raised
    assert seen == {"model": "deepseek-coder:6.7b-instruct", "pattern": "cot", "mode": "pipeline"}


def test_cost_cap_skips_the_review_and_says_why(settings, store, monkeypatch):
    from docdebt.service import app as app_module

    store.log_usage(
        owner="acme",
        repo="demo",
        pr_number=1,
        kind="review",
        calls=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=2.0,
        latency_s=1.0,
    )
    store.save_settings({"max_total_cost_usd": 1.0})
    notices = []
    monkeypatch.setattr(
        app_module.ReviewFlow, "post_cost_cap_notice", lambda gh, ev, spent, cap: notices.append(cap)
    )
    monkeypatch.setattr(app_module, "_build_flow", lambda *a: pytest.fail("must not review over the cap"))

    app_module._handle_pull_request(settings, store, pr_event())
    assert notices == [1.0]


# -- on-demand scan ------------------------------------------------------------------


def test_store_allows_one_running_scan_per_repository(store):
    first = store.start_scan("acme", "demo")
    assert first is not None and store.start_scan("acme", "demo") is None
    assert store.start_scan("acme", "other") is not None  # another repository is independent
    store.finish_scan(first, prs=2, proposed=1, clean=1)
    scan = store.get_scan(first)
    assert (scan["status"], scan["prs"], scan["proposed"]) == ("done", 2, 1)
    assert store.start_scan("acme", "demo") is not None  # free again once finished


def test_scans_left_running_by_a_restart_are_closed(settings, store):
    scan_id = store.start_scan("acme", "demo")
    create_app(settings)  # a new process starting on the same database
    scan = store.get_scan(scan_id)
    assert scan["status"] == "failed" and "restarted" in scan["error"]


@pytest.fixture
def scan_calls(monkeypatch):
    from docdebt.service import app as app_module

    calls = []
    monkeypatch.setattr(app_module, "_run_scan", lambda *a: calls.append(a))
    return calls


def test_api_scan_starts_in_the_background_and_reports_its_state(client, store, scan_calls):
    store.connect_repository(owner="acme", repo="demo", installation_id="77")
    response = client.post("/api/repos/acme/demo/scan", headers=AUTH)
    assert response.status_code == 202 and response.json()["status"] == "running"
    scan_id = response.json()["id"]
    assert scan_calls and scan_calls[0][2:] == (scan_id, "acme", "demo", "77")

    assert client.post("/api/repos/acme/demo/scan", headers=AUTH).status_code == 409  # already running
    store.finish_scan(scan_id, prs=1, proposed=1)
    assert client.get(f"/api/scans/{scan_id}").json()["status"] == "done"
    assert client.get("/api/repos/acme/demo").json()["last_scan"]["id"] == scan_id
    assert client.get("/api/scans/999").status_code == 404


def test_api_scan_refusals(client, store, scan_calls):
    assert client.post("/api/repos/acme/demo/scan").status_code == 401  # no token
    assert client.post("/api/repos/acme/demo/scan", headers=AUTH).status_code == 409  # not connected
    store.connect_repository(owner="acme", repo="demo", installation_id="1")
    store.disconnect_repository(owner="acme", repo="demo")
    assert client.post("/api/repos/acme/demo/scan", headers=AUTH).status_code == 409  # removed since

    store.connect_repository(owner="acme", repo="demo", installation_id="1")
    store.log_usage(
        owner="acme",
        repo="demo",
        pr_number=1,
        kind="review",
        calls=1,
        input_tokens=1,
        output_tokens=1,
        cost_usd=3.0,
        latency_s=1.0,
    )
    store.save_settings({"max_total_cost_usd": 1.0})
    response = client.post("/api/repos/acme/demo/scan", headers=AUTH)
    assert response.status_code == 409 and "cost cap" in response.json()["detail"]
    assert scan_calls == []


def test_run_scan_records_the_outcome_or_the_failure(settings, store, monkeypatch):
    from contextlib import contextmanager

    from docdebt.service import app as app_module
    from docdebt.service.review_flow import ScanOutcome

    class FakeFlow:
        def scan_repository(self, owner, repo, installation_id):
            return ScanOutcome(prs=2, proposed=1, clean=1)

    @contextmanager
    def fake_build_flow(*args):
        yield FakeFlow()

    monkeypatch.setattr(app_module, "_build_flow", fake_build_flow)
    ok = store.start_scan("acme", "demo")
    app_module._run_scan(settings, store, ok, "acme", "demo", "1")
    assert (store.get_scan(ok)["status"], store.get_scan(ok)["proposed"]) == ("done", 1)

    @contextmanager
    def broken_build_flow(*args):
        raise RuntimeError("GitHub unreachable")
        yield

    monkeypatch.setattr(app_module, "_build_flow", broken_build_flow)
    failed = store.start_scan("acme", "demo")
    app_module._run_scan(settings, store, failed, "acme", "demo", "1")
    assert store.get_scan(failed)["status"] == "failed" and "GitHub unreachable" in store.get_scan(failed)["error"]


# -- installations synced from GitHub ---------------------------------------------------


def fake_github(installations, *, fail=False):
    """A GitHub that knows `installations`: {installation_id: ["owner/repo", ...]}."""
    import httpx

    def handler(request):
        path = request.url.path
        if fail:
            return httpx.Response(401, json={"message": "Bad credentials"})
        if path == "/app/installations":
            return httpx.Response(200, json=[{"id": int(i)} for i in installations])
        if path.endswith("/access_tokens"):
            installation_id = path.split("/")[-2]
            return httpx.Response(
                201, json={"token": f"tok-{installation_id}", "expires_at": "2099-01-01T00:00:00Z"}
            )
        if path == "/installation/repositories":
            installation_id = request.headers["authorization"].removeprefix("Bearer tok-")
            repos = [
                {"name": name.split("/")[1], "owner": {"login": name.split("/")[0]}}
                for name in installations[installation_id]
            ]
            return httpx.Response(200, json={"repositories": repos})
        raise AssertionError(f"unexpected {request.method} {path}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sync_connects_what_github_lists_and_disconnects_the_rest(settings, store, monkeypatch):
    from docdebt.service.app import sync_installations

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "app.jwt")
    store.connect_repository(owner="acme", repo="uninstalled", installation_id="1")
    add(store, owner="NASSERamine", repo="docdebt-test-repo")  # activity, but no installation row

    result = sync_installations(
        settings, store, http=fake_github({"10": ["NASSERamine/docdebt-test-repo"], "20": ["acme/demo"]})
    )

    assert result == {
        "connected": ["NASSERamine/docdebt-test-repo", "acme/demo"],
        "disconnected": ["acme/uninstalled"],
    }
    rows = {r["repo"]: r for r in store.installations()}
    assert rows["docdebt-test-repo"]["active"] and rows["docdebt-test-repo"]["installation_id"] == "10"
    assert rows["demo"]["installation_id"] == "20" and not rows["uninstalled"]["active"]


def test_sync_that_fails_changes_nothing(settings, store, monkeypatch):
    from docdebt.service.app import sync_installations
    from docdebt.service.github_client import GithubApiError

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "app.jwt")
    store.connect_repository(owner="acme", repo="demo", installation_id="1")
    with pytest.raises(GithubApiError):
        sync_installations(settings, store, http=fake_github({}, fail=True))
    assert store.installations()[0]["active"]  # not disconnected because GitHub was unreachable


def test_api_sync_needs_the_token_and_reports_github_errors(client, monkeypatch):
    from docdebt.service import app as app_module
    from docdebt.service.github_client import GithubApiError

    assert client.post("/api/installations/sync").status_code == 401
    monkeypatch.setattr(app_module, "sync_installations", lambda s, st: {"connected": ["a/b"], "disconnected": []})
    assert client.post("/api/installations/sync", headers=AUTH).json()["connected"] == ["a/b"]

    def refused(s, st):
        import httpx

        raise GithubApiError("GET", "/app/installations", httpx.Response(401, text="Bad credentials"))

    monkeypatch.setattr(app_module, "sync_installations", refused)
    response = client.post("/api/installations/sync", headers=AUTH)
    assert response.status_code == 502 and "401" in response.json()["detail"]


def test_startup_sync_runs_only_when_enabled(settings, monkeypatch):
    from dataclasses import replace

    from docdebt.service import app as app_module

    calls = []
    monkeypatch.setattr(app_module, "_sync_installations_logged", lambda s, st: calls.append(s))
    create_app(settings)
    assert calls == []
    create_app(replace(settings, sync_installations_on_startup=True))
    import time as _time

    _time.sleep(0.2)  # the sync runs in a thread
    assert len(calls) == 1


def test_a_scans_table_from_the_previous_version_gains_the_new_column(settings):
    import sqlite3

    conn = sqlite3.connect(settings.state_db_path)
    conn.execute(
        "CREATE TABLE scans (id INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, repo TEXT NOT NULL, "
        "status TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL, prs INTEGER NOT NULL DEFAULT 0, "
        "proposed INTEGER NOT NULL DEFAULT 0, already_proposed INTEGER NOT NULL DEFAULT 0, "
        "clean INTEGER NOT NULL DEFAULT 0, not_proposed INTEGER NOT NULL DEFAULT 0, "
        "failed_prs INTEGER NOT NULL DEFAULT 0, error TEXT)"
    )
    conn.execute("INSERT INTO scans (owner, repo, status, started_at) VALUES ('acme', 'demo', 'done', 0)")
    conn.commit()
    conn.close()

    store = ReviewStore(settings.state_db_path)
    assert store.get_scan(1)["already_decided"] == 0
    scan_id = store.start_scan("acme", "demo")
    store.finish_scan(scan_id, prs=1, already_decided=1)
    assert store.get_scan(scan_id)["already_decided"] == 1
