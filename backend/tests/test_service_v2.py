"""Service v2: repository policy, learning from decisions, check runs, the
local -> cloud cascade, the documentation health scan and their API routes."""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from docdebt.core.auditor import Auditor
from docdebt.graph.build import GraphApp
from docdebt.graph.nodes import GraphDeps
from docdebt.llm.client import LLMClient
from docdebt.llm.config import ModelSpec
from docdebt.llm.prompts.registry import LOCKED_DETECTION_WORDINGS
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.rag.retriever import RagProvider
from docdebt.repair.hybrid_repairer import HybridRepairer
from docdebt.repair.repairer import Repairer
from docdebt.service.app import create_app
from docdebt.service.commands import parse_slash_command
from docdebt.service.config import ServiceSettings
from docdebt.service.engine import build_service_deps
from docdebt.service.github_client import GithubClient
from docdebt.service.health import measure, run_health_scan
from docdebt.service.learning import precision, threshold_suggestion
from docdebt.service.review_flow import ReviewFlow, check_run_fields
from docdebt.service.runtime_settings import default_runtime_settings
from docdebt.service.store import ReviewStore

from .conftest import make_factory
from .test_service_review_flow import SOURCE, FakeGithub, pr_event

TOKEN = "dash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def settings(tmp_path):
    return ServiceSettings(
        github_app_id="1",
        github_app_slug="app",
        github_private_key="k",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
        dashboard_token=TOKEN,
        cloud_model="gpt-5.6-luna",
        cloud_price_in_per_m=1.25,
        cloud_price_out_per_m=10.0,
    )


@pytest.fixture
def store(settings):
    return ReviewStore(settings.state_db_path)


def add(store, label="X.f", **metadata):
    return store.create(
        thread_id=f"t-{label}",
        owner="acme",
        repo="demo",
        pr_number=1,
        file_path="X.java",
        method_label=label,
        head_sha="sha",
        installation_id="1",
        metadata=metadata,
    )


# -- repository policy ------------------------------------------------------------------


def test_repo_settings_have_safe_defaults_and_are_saved(store):
    assert store.repo_settings("acme", "demo") == {
        "allow_cloud": False,
        "block_on_structural": False,
        "rag_enabled": True,
        "confidence_threshold": None,
        "updated_at": None,
    }
    saved = store.save_repo_settings("acme", "demo", {"allow_cloud": True, "confidence_threshold": 0.75})
    assert saved["allow_cloud"] and saved["confidence_threshold"] == 0.75 and saved["rag_enabled"]
    saved = store.save_repo_settings("acme", "demo", {"block_on_structural": True})
    assert saved["allow_cloud"] and saved["block_on_structural"]  # other values kept


# -- learning from decisions -----------------------------------------------------------


def test_metadata_and_reject_reason_are_stored_with_the_proposal(store):
    sid = add(
        store, debt_kind="semantic", rules="semantic", confidence=0.66, detection_model="gpt-5.6-luna", line=7
    )
    store.mark_resolved(sid, "reject", "false_positive")
    r = store.get(sid)
    assert (r.debt_kind, r.confidence, r.detection_model, r.line) == ("semantic", 0.66, "gpt-5.6-luna", 7)
    assert r.reject_reason == "false_positive" and r.decided_at is not None
    with pytest.raises(ValueError):
        store.mark_resolved(add(store, "Y.g"), "reject", "because")


def test_an_older_database_gains_the_new_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE review_requests (short_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, "
            "owner TEXT NOT NULL, "
            "repo TEXT NOT NULL, pr_number INTEGER NOT NULL, file_path TEXT NOT NULL, method_label TEXT NOT NULL, "
            "head_sha TEXT NOT NULL, resolved INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO review_requests VALUES ('abc12345', 't', 'a', 'b', 1, 'X.java', 'X.f', 's', 0, 1)"
        )
    store = ReviewStore(path)
    assert store.get("abc12345").debt_kind is None  # kept, readable


def test_precision_and_its_breakdown(store):
    for i in range(8):
        store.mark_resolved(add(store, f"A.s{i}", debt_kind="structural", repair_mechanism="hybrid"), "accept")
    for i, reason in enumerate(["false_positive", "manual"]):
        store.mark_resolved(
            add(store, f"B.s{i}", debt_kind="semantic", repair_mechanism="hybrid"), "reject", reason
        )
    store.mark_resolved(add(store, "C.e"), "expired")  # nobody decided: excluded
    p = precision(store.decided_proposals())
    assert (p["decisions"], p["accepted"], p["precision"], p["enough"]) == (10, 8, 0.8, True)
    kinds = {row["key"]: row["precision"] for row in p["by_kind"]}
    assert kinds == {"structural": 1.0, "semantic": 0.0}
    assert p["reject_reasons"] == {"false_positive": 1, "manual": 1}


def test_threshold_suggestion_picks_the_lowest_threshold_reaching_the_target(store):
    # 10 confident accepted, 10 low-confidence: mostly rejected
    for i in range(10):
        store.mark_resolved(add(store, f"H.{i}", debt_kind="semantic", confidence=1.0), "accept")
    for i in range(10):
        sid = add(store, f"L.{i}", debt_kind="semantic", confidence=0.6)
        store.mark_resolved(sid, "accept" if i < 2 else "reject")
    s = threshold_suggestion(store.decided_proposals(), target=0.8, current=0.0)
    assert s["enough"] and s["decisions"] == 20
    assert [row["threshold"] for row in s["curve"]] == [0.6, 1.0]
    assert s["curve"][0]["precision"] == 0.6 and s["curve"][1]["precision"] == 1.0
    assert s["suggested"] == 1.0
    few = threshold_suggestion(store.decided_proposals()[:5])
    assert not few["enough"] and few["suggested"] is None


def test_reject_reasons_in_slash_commands():
    assert parse_slash_command("/docdebt reject 1a2b3c4d false-positive").reason == "false_positive"
    assert parse_slash_command("/docdebt reject 1a2b3c4d faux-positif").reason == "false_positive"
    assert parse_slash_command("/docdebt reject 1a2b3c4d").reason is None
    assert parse_slash_command("/docdebt reject 1a2b3c4d whatever").reason is None
    assert parse_slash_command("/docdebt accept 1a2b3c4d manual").reason is None  # accepts have no reason


# -- check runs ------------------------------------------------------------------------


def test_check_run_conclusions(store):
    assert check_run_fields([], block_on_structural=True)["conclusion"] == "success"
    structural = store.get(add(store, "S.f", debt_kind="structural", rules="missing-param", line=4))
    semantic = store.get(add(store, "M.f", debt_kind="semantic", line=9))
    fields = check_run_fields([structural, semantic], block_on_structural=False, public_url="http://x")
    assert fields["conclusion"] == "neutral" and "2 documentation fixes" in fields["output"]["title"]
    assert "http://x/dashboard/proposals/" in fields["output"]["text"]
    assert [a["start_line"] for a in fields["output"]["annotations"]] == [4, 9]
    blocked = check_run_fields([structural, semantic], block_on_structural=True)
    assert blocked["conclusion"] == "failure"
    assert [a["annotation_level"] for a in blocked["output"]["annotations"]] == ["failure", "warning"]
    assert check_run_fields([semantic], block_on_structural=True)["conclusion"] == "neutral"


class ChecksGithub(FakeGithub):
    """FakeGithub that also serves check runs (or refuses them with 403)."""

    def __init__(self, checks_status=200, **kw):
        super().__init__(**kw)
        self.checks_status = checks_status
        self.check_runs: list[dict] = []

    def handler(self, request):
        path, method = request.url.path, request.method
        if "/check-runs" in path:
            if self.checks_status != 200:
                return httpx.Response(self.checks_status, json={"message": "Resource not accessible"})
            if method == "GET":
                return httpx.Response(200, json={"check_runs": [{"id": 1}] if self.check_runs else []})
            self.check_runs.append({"method": method, **json.loads(request.content)})
            return httpx.Response(201 if method == "POST" else 200, json={"id": 1})
        return super().handler(request)


def hybrid_flow(monkeypatch, fake, tmp_path, answer, *, knowledge=None, rag=None):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    github = GithubClient(
        "app-1", "k", "install-1", http=httpx.Client(transport=httpx.MockTransport(fake.handler))
    )
    spec = ModelSpec(
        id="qwen2.5-coder:7b", provider="ollama", model="qwen2.5-coder:7b", max_tokens={"repair": 900}
    )
    client = LLMClient(spec, factory=make_factory(lambda p, m: (json.dumps(answer), "stop")), sleep=lambda s: None)
    repairer = HybridRepairer(client)
    deps = GraphDeps(auditor=Auditor(), router=None, repairer=repairer, rag=rag, rag_repair=rag is not None)
    store = ReviewStore(tmp_path / "state.sqlite3")
    return ReviewFlow(
        github, GraphApp(deps, InMemorySaver()), store, knowledge=knowledge, public_url="http://x", check_runs=True
    )


def test_a_review_stores_metadata_publishes_a_check_run_and_accepting_teaches_the_knowledge_base(
    monkeypatch, tmp_path
):
    fake = ChecksGithub()
    kb = KnowledgeBase(tmp_path / "state.sqlite3")
    rag = RagProvider(kb, "acme", "demo", sources=("siblings", "similar"))
    flow = hybrid_flow(
        monkeypatch, fake, tmp_path, {"m": {"description": "keep", "@return": "keep"}}, knowledge=kb, rag=rag
    )
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    assert (proposal.debt_kind, proposal.repair_mechanism, proposal.line) == ("structural", "hybrid", 5)
    assert "missing-param" in proposal.rules and "unknown-param" in proposal.rules
    assert fake.check_runs[0]["method"] == "POST" and fake.check_runs[0]["conclusion"] == "neutral"
    state = flow.graph.get_state(proposal.thread_id)
    assert state["repair_mechanism"] == "hybrid" and state["field_log"]

    outcome = flow.decide(proposal.short_id, "accept", owner="acme", repo="demo", pr_number=1)
    assert (
        outcome.status.value == "applied"
        and "tempInKelvin" in base64.b64decode(fake.commits[0]["content"]).decode()
    )
    assert fake.check_runs[-1]["method"] == "PATCH" and fake.check_runs[-1]["conclusion"] == "success"
    assert kb.stats("acme", "demo")["counts"]["validated"] == 1


def test_a_missing_checks_permission_never_breaks_a_review(monkeypatch, tmp_path, caplog):
    fake = ChecksGithub(checks_status=403)
    flow = hybrid_flow(monkeypatch, fake, tmp_path, {"m": {"description": "keep", "@return": "keep"}})
    flow.github.installation_id = "install-403"
    assert flow.review_pull_request(pr_event()).proposed == 1
    assert "Checks: Read and" in caplog.text


def test_a_reject_with_a_reason_is_recorded(monkeypatch, tmp_path):
    fake = ChecksGithub()
    flow = hybrid_flow(monkeypatch, fake, tmp_path, {"m": {"description": "keep", "@return": "keep"}})
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    flow.decide(proposal.short_id, "reject", owner="acme", repo="demo", pr_number=1, reason="incorrect_fix")
    assert flow.store.get(proposal.short_id).reject_reason == "incorrect_fix"


# -- the local -> cloud cascade ------------------------------------------------------


def test_luna_reuses_the_gpt_5_6_detection_wordings():
    assert LOCKED_DETECTION_WORDINGS["gpt-5.6-luna"] == LOCKED_DETECTION_WORDINGS["gpt-5.6"]


def test_the_cloud_model_joins_the_cascade_only_when_the_repository_allows_it(settings):
    runtime = default_runtime_settings(settings).model_copy(update={"structural_only": False})
    local_only = build_service_deps(settings, runtime, {"allow_cloud": False}, knowledge=None)
    assert local_only.router.policy.cascade == ["qwen2.5-coder:7b"] and not local_only.router.policy.allow_cloud
    both = build_service_deps(settings, runtime, {"allow_cloud": True, "confidence_threshold": 0.7})
    assert both.router.policy.cascade == ["qwen2.5-coder:7b", "gpt-5.6-luna"]
    assert both.router.judges["gpt-5.6-luna"].config.confidence_threshold == 0.7
    luna = both.repairers["gpt-5.6-luna"]
    assert isinstance(luna, HybridRepairer)  # no screened regeneration wording: always hybrid
    assert luna.client.spec.usd_per_1k_input == pytest.approx(0.00125)  # priced from the environment


def test_the_mechanism_setting_chooses_the_local_repairer(settings):
    hybrid = default_runtime_settings(settings)
    regen = hybrid.model_copy(update={"repair_mechanism": "regeneration"})
    assert isinstance(build_service_deps(settings, hybrid, {}).repairer, HybridRepairer)
    assert isinstance(build_service_deps(settings, regen, {}).repairer, Repairer)
    structural = build_service_deps(settings, hybrid, {"allow_cloud": True})
    assert structural.router is None  # structural only: no Judge, so no cloud call either


def test_rag_follows_the_repository_policy(settings):
    runtime = default_runtime_settings(settings)
    on = build_service_deps(settings, runtime, {"rag_enabled": True}, owner="acme", repo="demo", embedder=False)
    off = build_service_deps(settings, runtime, {"rag_enabled": False}, owner="acme", repo="demo")
    assert on.rag is not None and on.rag_repair and not on.rag_detection
    assert off.rag is None and not off.rag_repair


# -- documentation health ---------------------------------------------------------------


def test_measure_counts_methods_documentation_and_debt():
    figures = measure({"src/main/java/demo/Temperature.java": SOURCE})
    assert (figures["methods"], figures["documented"], figures["debt_methods"]) == (1, 1, 1)
    assert figures["score"] == 0.0 and figures["packages"][0]["package"] == "demo"
    assert figures["worst_files"][0]["path"].endswith("Temperature.java")


class RepoGithub(FakeGithub):
    """Serves a small repository for the health scan."""

    FILES = {
        "src/main/java/demo/Temperature.java": SOURCE,
        "src/main/java/demo/Ok.java": (
            "package demo;\nclass Ok {\n /**\n  * Says hi.\n  */\n public void hi() {}\n}\n"
        ),
        "src/test/java/demo/TemperatureTest.java": "class TemperatureTest { void t() { x.toCelsius(1); } }",
        "README.md": "# Demo\nTemperatures in kelvin.",
    }

    def handler(self, request):
        path = request.url.path
        if path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        if path == "/repos/acme/demo":
            return httpx.Response(200, json={"default_branch": "main"})
        if path == "/repos/acme/demo/commits/main":
            return httpx.Response(200, json={"sha": "head1"})
        if path == "/repos/acme/demo/git/trees/head1":
            return httpx.Response(200, json={"tree": [{"path": p, "type": "blob"} for p in self.FILES]})
        if "/contents/" in path:
            name = path.split("/contents/", 1)[1]
            return httpx.Response(
                200, json={"content": base64.b64encode(self.FILES[name].encode()).decode(), "sha": "x"}
            )
        raise AssertionError(f"unexpected call: {request.method} {path}")


def test_health_scan_measures_the_default_branch_and_builds_the_knowledge_base(monkeypatch, tmp_path):
    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    fake = RepoGithub()
    github = GithubClient("1", "k", "9", http=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    values = run_health_scan(github, "acme", "demo", knowledge=kb)
    assert (values["branch"], values["commit_sha"], values["files"]) == ("main", "head1", 2)
    assert (values["methods"], values["documented"], values["debt_methods"]) == (2, 2, 1)
    assert values["score"] == 0.5 and not values["truncated"]
    counts = kb.stats("acme", "demo")["counts"]
    assert counts == {"method": 1, "test": 1, "readme": 1, "validated": 0}
    limited = run_health_scan(github, "acme", "demo", max_files=1)
    assert limited["truncated"] and limited["files"] == 1


# -- API routes --------------------------------------------------------------------------


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def test_repo_settings_routes(client, store):
    add(store)  # the repository is known
    assert client.get("/api/repos/acme/demo/settings").json()["allow_cloud"] is False
    assert client.put("/api/repos/acme/demo/settings", json={"allow_cloud": True}).status_code == 401  # no token
    body = client.put(
        "/api/repos/acme/demo/settings", json={"allow_cloud": True, "confidence_threshold": 0.8}, headers=AUTH
    ).json()
    assert body["allow_cloud"] and body["confidence_threshold"] == 0.8 and body["cloud_model"] == "gpt-5.6-luna"
    cleared = client.put("/api/repos/acme/demo/settings", json={"clear_threshold": True}, headers=AUTH).json()
    assert cleared["confidence_threshold"] is None
    assert (
        client.put("/api/repos/acme/demo/settings", json={"confidence_threshold": 2}, headers=AUTH).status_code
        == 422
    )


def test_health_routes_run_the_scan_in_the_background(client, store, monkeypatch, settings):
    from docdebt.service import app as app_module

    store.connect_repository(owner="acme", repo="demo", installation_id="9")
    monkeypatch.setattr(
        app_module,
        "run_health_scan",
        lambda *a, **k: {
            "branch": "main",
            "commit_sha": "h",
            "files": 3,
            "methods": 10,
            "documented": 8,
            "debt_methods": 2,
            "warnings": 1,
            "score": 0.6,
            "truncated": False,
            "packages": [{"package": "demo", "score": 0.6}],
            "worst_files": [],
        },  # fmt: skip
    )
    started = client.post("/api/repos/acme/demo/health-scan", headers=AUTH)
    assert started.status_code == 202
    health = client.get("/api/repos/acme/demo/health").json()
    assert health["latest"]["score"] == 0.6 and health["trend"][0]["score"] == 0.6
    assert health["latest"]["packages"] == [{"package": "demo", "score": 0.6}]
    assert client.get(f"/api/health-scans/{started.json()['id']}").json()["status"] == "done"
    assert client.post("/api/repos/acme/other/health-scan", headers=AUTH).status_code == 409  # not connected


def test_knowledge_precision_and_threshold_routes(client, store):
    for i in range(3):
        store.mark_resolved(add(store, f"A.{i}", debt_kind="semantic", confidence=0.9), "accept")
    store.mark_resolved(add(store, "B.0", debt_kind="semantic", confidence=0.5), "reject", "false_positive")
    knowledge = client.get("/api/repos/acme/demo/knowledge").json()
    assert knowledge["total"] == 0 and knowledge["embedder"] == "bm25"
    p = client.get("/api/precision?repo=acme/demo").json()
    assert p["decisions"] == 4 and p["precision"] == 0.75 and not p["enough"]
    t = client.get("/api/repos/acme/demo/threshold-suggestion").json()
    assert t["decisions"] == 4 and t["current"] == 0.0 and not t["enough"]


def test_settings_expose_the_cloud_model_and_new_choices(client):
    body = client.get("/api/settings").json()
    assert body["cloud"]["model"] == "gpt-5.6-luna" and body["cloud"]["known"]
    assert body["choices"]["mechanisms"] == ["hybrid", "regeneration"]
    assert body["effective"]["repair_mechanism"] == "hybrid" and body["effective"]["rag_repair"] is True
    saved = client.put(
        "/api/settings",
        json={**body["effective"], "rag_sources": ["siblings", "readme"], "rag_top_k": 3},
        headers=AUTH,
    ).json()
    assert saved["effective"]["rag_sources"] == ["siblings", "readme"] and saved["effective"]["rag_top_k"] == 3


def test_dashboard_decision_passes_the_reject_reason(client, store, monkeypatch):
    from docdebt.service import app as app_module
    from docdebt.service.review_flow import DecisionOutcome, DecisionStatus

    seen = {}

    def fake(settings, store_, request, decision, installation_id, reason=None):
        seen.update(decision=decision, reason=reason)
        return DecisionOutcome(DecisionStatus.REJECTED, "rejected")

    monkeypatch.setattr(app_module, "_decide_from_dashboard", fake)
    sid = add(store)
    response = client.post(
        f"/api/proposals/{sid}/decide", json={"decision": "reject", "reason": "manual"}, headers=AUTH
    )
    assert response.status_code == 200 and seen == {"decision": "reject", "reason": "manual"}
    bad = client.post(f"/api/proposals/{sid}/decide", json={"decision": "reject", "reason": "lazy"}, headers=AUTH)
    assert bad.status_code == 422
