"""Service v3: the model chosen in the application, one detection on the whole
Javadoc (exact rules OR the model), retrieval always on, Replacement first and
Regeneration on demand, accept with an edit, costs, the push-driven index, the
status banner and the review queue."""

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from docdebt.graph.build import GraphApp
from docdebt.llm.catalog import DEFAULT_SERVICE_MODEL, SERVICE_MODELS, service_model_choices
from docdebt.llm.prompts.registry import LOCKED_DETECTION_WORDINGS, is_screened
from docdebt.rag.knowledge import KnowledgeBase
from docdebt.repair.hybrid_repairer import HybridRepairer
from docdebt.service import engine
from docdebt.service.app import create_app
from docdebt.service.commands import parse_slash_command
from docdebt.service.config import ServiceSettings
from docdebt.service.engine import build_service_deps
from docdebt.service.github_client import GithubClient
from docdebt.service.health import measure, run_health_scan, run_push_index
from docdebt.service.learning import precision
from docdebt.service.llm_settings import LOCAL_SCOPE, build_spec, resolve_model, scope_for_repo
from docdebt.service.review_flow import (
    DecisionStatus,
    ReviewFlow,
    check_run_fields,
    edited_javadoc_problems,
    normalize_edited_javadoc,
)
from docdebt.service.secrets import decrypt, encrypt, key_hint, load_cipher
from docdebt.service.store import ReviewStore
from docdebt.service.webhooks import parse_push_event

from .conftest import KELVIN_SOURCE, make_factory
from .test_service_review_flow import SOURCE, FakeGithub, pr_event

TOKEN = "dash-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
KEY = "sk-test-0123456789abcd"


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCDEBT_SECRET_KEY", raising=False)
    return ServiceSettings(
        github_app_id="1",
        github_app_slug="app",
        github_private_key="k",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
        dashboard_token=TOKEN,
        rag_embeddings="bm25",
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


# -- a scripted provider ------------------------------------------------------------------

FIXED = "/**\n     * Converts kelvin to celsius.\n     * @param tempInKelvin temperature in kelvin\n"


def responder(*, verdict="INCOMPLETE", explain="NONE", hybrid=None, regen=None, fail=False):
    """Answers each prompt of the pipeline by its kind."""

    def answer(prompt, max_tokens):
        if fail:
            return RuntimeError("401 invalid api key")
        if "Reply with the single word OK" in prompt:
            return ("OK", "stop")
        if "COMPLETE or INCOMPLETE" in prompt:
            return (verdict, "stop")
        if "CLAIM:" in prompt and "FIELD:" in prompt:
            return (explain, "stop")
        if "FIELDS TO" in prompt:
            return (json.dumps(hybrid or {"m": {"description": "keep", "@return": "keep"}}), "stop")
        return (regen or "no javadoc", "stop")

    return answer


# -- the model configuration ------------------------------------------------------------


def test_the_application_offers_the_four_research_api_models_luna_first():
    assert SERVICE_MODELS == ("gpt-5.6-luna", "gpt-5.6", "claude-sonnet-5", "gemini-3.1-pro")
    assert DEFAULT_SERVICE_MODEL == "gpt-5.6-luna"
    choices = service_model_choices()
    assert [c["provider"] for c in choices] == ["openai", "openai", "anthropic", "google"]
    assert [c["default"] for c in choices] == [True, False, False, False]
    assert all(m in LOCKED_DETECTION_WORDINGS for m in SERVICE_MODELS)  # every model has a fixed prompt
    assert LOCKED_DETECTION_WORDINGS["gpt-5.6-luna"] == LOCKED_DETECTION_WORDINGS["gpt-5.6"]
    assert is_screened("gpt-5.6-luna") and not is_screened("gemini-3.1-pro")


def test_keys_are_encrypted_and_only_their_hint_is_kept(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCDEBT_SECRET_KEY", raising=False)
    cipher = load_cipher(tmp_path)
    token = encrypt(cipher, KEY)
    assert KEY not in token and decrypt(load_cipher(tmp_path), token) == KEY  # same key file reused
    assert (tmp_path / "secret.key").is_file() and key_hint(KEY) == "…abcd"
    monkeypatch.setenv("DOCDEBT_SECRET_KEY", "another secret")
    with pytest.raises(Exception, match="type it again"):
        decrypt(load_cipher(tmp_path), token)


def test_a_spec_carries_the_key_and_the_prices():
    spec = build_spec("claude-sonnet-5", KEY, price_in_per_m=3.0, price_out_per_m=15.0)
    assert (spec.provider, spec.api_key, spec.api_key_env) == ("anthropic", KEY, None)
    assert spec.cost_usd(1000, 1000) == pytest.approx(0.018)
    assert KEY not in spec.model_dump_json() and KEY not in repr(spec)
    with pytest.raises(ValueError):
        build_spec("qwen2.5-coder:7b", KEY)  # local models are not offered


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(engine, "CHAT_FACTORY", make_factory(responder()))
    return TestClient(create_app(settings))


def test_saving_tests_the_connection_and_never_returns_the_key(client, settings, monkeypatch):
    assert client.get("/api/llm-config").json()["config"] is None
    body = {"model": "gpt-5.6-luna", "api_key": KEY, "price_in_per_m": 1.25, "price_out_per_m": 10}
    assert client.put("/api/llm-config", json=body).status_code == 401  # needs the token
    saved = client.put("/api/llm-config", json=body, headers=AUTH)
    assert saved.status_code == 200 and KEY not in saved.text
    config = saved.json()["config"]
    assert (config["model"], config["key_hint"], config["last_test_ok"]) == ("gpt-5.6-luna", "…abcd", True)
    # Kept across a restart; changing only the model reuses the saved key.
    again = TestClient(create_app(settings))
    changed = again.put("/api/llm-config", json={"model": "gemini-3.1-pro"}, headers=AUTH).json()["config"]
    assert changed["model"] == "gemini-3.1-pro" and changed["key_hint"] == "…abcd"
    assert KEY not in again.get("/api/llm-config").text
    row = ReviewStore(settings.state_db_path).llm_config(LOCAL_SCOPE)
    assert KEY not in row["api_key_enc"]


def test_a_failed_test_saves_nothing(client, monkeypatch, settings):
    monkeypatch.setattr(engine, "CHAT_FACTORY", make_factory(responder(fail=True)))
    test = client.post("/api/llm-config/test", json={"model": "gpt-5.6", "api_key": KEY}, headers=AUTH).json()
    assert test["ok"] is False and "invalid api key" in test["error"]
    refused = client.put("/api/llm-config", json={"model": "gpt-5.6", "api_key": KEY}, headers=AUTH)
    assert refused.status_code == 422 and "invalid api key" in refused.json()["detail"]
    assert client.get("/api/llm-config").json()["config"] is None
    assert client.put("/api/llm-config", json={"model": "gpt-9", "api_key": KEY}, headers=AUTH).status_code == 422
    assert client.put("/api/llm-config", json={"model": "gpt-5.6"}, headers=AUTH).status_code == 422  # no key


def test_deleting_the_configuration(client):
    client.put("/api/llm-config", json={"model": "gpt-5.6-luna", "api_key": KEY}, headers=AUTH)
    assert client.delete("/api/llm-config", headers=AUTH).json()["config"] is None


def test_a_repository_is_reviewed_with_the_configuration_of_its_first_user(store, settings):
    cipher = load_cipher(settings.state_db_path.parent)
    for github_id in (2, 1):
        store.upsert_user(github_id=github_id, login=f"u{github_id}", name=None, avatar_url=None)
    store.add_user_repo(1, "acme", "demo")
    store.add_user_repo(2, "acme", "demo")
    assert scope_for_repo(store, auth_enabled=True, owner="acme", repo="demo") is None
    store.save_llm_config(
        "user:2", model="claude-sonnet-5", api_key_enc=encrypt(cipher, KEY), key_hint="…", price_in_per_m=None,
        price_out_per_m=None,
    )  # fmt: skip
    assert scope_for_repo(store, auth_enabled=True, owner="acme", repo="demo") == "user:2"
    resolved = resolve_model(cipher, store, "user:2")
    assert resolved.spec.id == "claude-sonnet-5" and resolved.spec.api_key == KEY
    assert scope_for_repo(store, auth_enabled=False, owner="acme", repo="demo") is None  # local: none saved


# -- deterministic engine -------------------------------------------------------------------


def test_without_a_model_only_the_exact_rules_run():
    deps = build_service_deps(None, owner="acme", repo="demo")
    assert deps.router is None and deps.repairer is None and deps.rag is not None


def test_one_model_one_vote_no_threshold_context_always_on():
    spec = build_spec("gpt-5.6-luna", KEY)
    deps = build_service_deps(spec, owner="acme", repo="demo")
    assert deps.router.policy.cascade == ["gpt-5.6-luna"]
    judge = deps.router.judges["gpt-5.6-luna"]
    cfg = judge.config
    assert (cfg.patterns, cfg.samples_per_pattern, cfg.confidence_threshold) == (["zeroshot"], 1, 0.0)
    assert cfg.use_context and cfg.keep_unverified_explanations and not cfg.require_citation
    assert judge.client.spec.temperature == 0.0
    assert isinstance(deps.repairer, HybridRepairer) and deps.rag_repair and deps.rag_detection


# -- one detection on the whole Javadoc -------------------------------------------------------


class ChecksGithub(FakeGithub):
    """FakeGithub that also serves check runs (or refuses them with 403) and any diff."""

    def __init__(self, checks_status=200, diff=None, **kw):
        super().__init__(**kw)
        self.checks_status = checks_status
        self.check_runs: list[dict] = []
        self.diff = diff

    def handler(self, request):
        path, method = request.url.path, request.method
        if "/check-runs" in path:
            if self.checks_status != 200:
                return httpx.Response(self.checks_status, json={"message": "Resource not accessible"})
            if method == "GET":
                return httpx.Response(200, json={"check_runs": [{"id": 1}] if self.check_runs else []})
            self.check_runs.append({"method": method, **json.loads(request.content)})
            return httpx.Response(201 if method == "POST" else 200, json={"id": 1})
        if self.diff and "/pulls/" in path and request.headers.get("accept", "").endswith("v3.diff"):
            return httpx.Response(200, text=self.diff)
        return super().handler(request)


def flow_for(monkeypatch, tmp_path, fake, answer, *, prices=False, knowledge=None):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    monkeypatch.setattr(engine, "CHAT_FACTORY", make_factory(answer))
    github = GithubClient(
        "app-1", "k", "install-1", http=httpx.Client(transport=httpx.MockTransport(fake.handler))
    )
    spec = build_spec(
        "gpt-5.6-luna", KEY, **({"price_in_per_m": 1000.0, "price_out_per_m": 1000.0} if prices else {})
    )
    spec.max_retries = 1
    from docdebt.llm.client import UsageTracker

    tracker = UsageTracker()
    deps = build_service_deps(spec, owner="acme", repo="demo", tracker=tracker, knowledge=knowledge)
    store = ReviewStore(tmp_path / "state.sqlite3")
    store.save_llm_config(
        LOCAL_SCOPE, model=spec.id, api_key_enc="x", key_hint="…", price_in_per_m=None, price_out_per_m=None
    )
    return ReviewFlow(
        github,
        GraphApp(deps, InMemorySaver()),
        store,
        tracker=tracker,
        knowledge=knowledge,
        public_url="http://x",
        check_runs=True,
        spec=spec,
        model_scope=LOCAL_SCOPE,
    )


PICK_DIFF = """diff --git a/src/Temperature.java b/src/Temperature.java
--- a/src/Temperature.java
+++ b/src/Temperature.java
@@ -20,1 +20,1 @@
-    public int pick(int a, int b) {
+    public int pick(int a, int b) {
"""


def test_an_exact_rule_is_debt_even_when_the_model_says_complete(monkeypatch, tmp_path):
    fake = ChecksGithub()
    flow = flow_for(monkeypatch, tmp_path, fake, responder(verdict="COMPLETE"))
    assert flow.review_pull_request(pr_event()).proposed == 1
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    sources = {item["source"] for item in proposal.explanation_items}
    assert "rule" in sources and "unknown-param" in proposal.rules
    assert proposal.repair_mechanism == "replacement"
    assert "/docdebt regenerate" in fake.comments[0] and "**Why**" in fake.comments[0]


def test_the_model_alone_can_find_debt_with_its_explanation(monkeypatch, tmp_path):
    fake = ChecksGithub(file_content=KELVIN_SOURCE, diff=PICK_DIFF)
    explanation = (
        "CLAIM: Returns the maximum of the two values.\nCODE: return Math.min(a, b);\n"
        "REASON: The code returns the minimum.\nFIELD: description\n---"
    )
    hybrid = {"m": {"description": "Returns the minimum of the two values.", "@return": "keep"}}
    flow = flow_for(monkeypatch, tmp_path, fake, responder(explain=explanation, hybrid=hybrid))
    assert flow.review_pull_request(pr_event()).proposed == 1
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    [item] = proposal.explanation_items
    assert item["source"] == "model" and item["verified"] and "minimum" in item["message"]
    assert "Math.min" in item["code"]


def test_no_rule_and_a_complete_verdict_proposes_nothing(monkeypatch, tmp_path):
    fake = ChecksGithub(file_content=KELVIN_SOURCE, diff=PICK_DIFF)
    flow = flow_for(monkeypatch, tmp_path, fake, responder(verdict="COMPLETE"))
    outcome = flow.review_pull_request(pr_event())
    assert outcome.proposed == 0 and outcome.skipped_no_debt == 1
    assert fake.check_runs[-1]["conclusion"] == "success"


def test_an_explanation_without_exact_quotes_is_kept_without_highlights(monkeypatch, tmp_path):
    fake = ChecksGithub(file_content=KELVIN_SOURCE, diff=PICK_DIFF)
    explanation = (
        "CLAIM: returns the largest value\nCODE: max(a, b)\nREASON: It is the minimum.\nFIELD: description\n---"
    )
    hybrid = {"m": {"description": "Returns the minimum of the two values.", "@return": "keep"}}
    flow = flow_for(monkeypatch, tmp_path, fake, responder(explain=explanation, hybrid=hybrid))
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    [item] = proposal.explanation_items
    assert item["message"] == "It is the minimum." and item["claim"] is None and not item["verified"]


def test_a_failing_model_leaves_the_exact_rules_and_says_so(monkeypatch, tmp_path):
    fake = ChecksGithub()
    flow = flow_for(monkeypatch, tmp_path, fake, responder(fail=True))
    outcome = flow.review_pull_request(pr_event())
    assert outcome.proposed == 0 and outcome.llm_errors == 1
    assert any("could not call the model" in c for c in fake.comments)
    row = flow.store.llm_config(LOCAL_SCOPE)
    assert row["last_test_ok"] == 0 and "invalid api key" in row["last_test_error"]


def test_the_cost_of_a_proposal_is_recorded(monkeypatch, tmp_path):
    fake = ChecksGithub()
    flow = flow_for(monkeypatch, tmp_path, fake, responder(), prices=True)
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    assert proposal.cost_usd > 0
    assert flow.store.pr_cost_usd("acme", "demo", 1) == pytest.approx(proposal.cost_usd)
    assert flow.store.monthly_costs()[-1]["cost_usd"] > 0


# -- check run --------------------------------------------------------------------------------


def test_check_run_conclusions(store):
    assert check_run_fields([], block_on_debt=True)["conclusion"] == "success"
    one = store.get(add(store, "S.f", rules="missing-param", line=4))
    two = store.get(add(store, "M.f", rules="contradicted-claim", line=9))
    fields = check_run_fields([one, two], block_on_debt=False, public_url="http://x")
    assert fields["conclusion"] == "neutral" and "2 documentation fixes" in fields["output"]["title"]
    assert "http://x/dashboard/proposals/" in fields["output"]["text"]
    blocked = check_run_fields([one, two], block_on_debt=True)
    assert blocked["conclusion"] == "failure"
    assert {a["annotation_level"] for a in blocked["output"]["annotations"]} == {"failure"}


def test_a_missing_checks_permission_never_breaks_a_review(monkeypatch, tmp_path, caplog):
    fake = ChecksGithub(checks_status=403)
    flow = flow_for(monkeypatch, tmp_path, fake, responder())
    flow.github.installation_id = "install-403"
    assert flow.review_pull_request(pr_event()).proposed == 1
    assert "Checks: Read and" in caplog.text


def test_repo_policy_is_only_the_merge_block(store):
    assert store.repo_settings("acme", "demo") == {"block_on_debt": False, "updated_at": None}
    assert store.save_repo_settings("acme", "demo", {"block_on_debt": True})["block_on_debt"] is True


# -- regeneration on demand, accept with an edit --------------------------------------------------


REGENERATED = (
    "/**\n * Converts a temperature from kelvin to celsius.\n * @param tempInKelvin the temperature, in kelvin\n"
    " * @return temperature in celsius\n */"
)


def test_regenerate_then_accept_the_regeneration(monkeypatch, tmp_path):
    fake = ChecksGithub()
    kb = KnowledgeBase(tmp_path / "state.sqlite3")
    flow = flow_for(monkeypatch, tmp_path, fake, responder(regen=REGENERATED), knowledge=kb)
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    outcome = flow.regenerate(proposal.short_id, owner="acme", repo="demo", pr_number=1)
    assert outcome.status == DecisionStatus.REGENERATED
    stored = flow.store.get(proposal.short_id)
    # Regeneration merges only the fields the findings plan to change (here the renamed @param).
    assert stored.regen_status == "verified" and "the temperature, in kelvin" in stored.regen_doc
    assert "regenerated" in fake.comments[-1]
    again = flow.regenerate(proposal.short_id, owner="acme", repo="demo", pr_number=1)
    assert again.status == DecisionStatus.REGENERATED  # no second model call
    applied = flow.decide(
        proposal.short_id, "accept", owner="acme", repo="demo", pr_number=1, mechanism="regeneration"
    )
    assert applied.status == DecisionStatus.APPLIED
    committed = base64.b64decode(fake.commits[0]["content"]).decode()
    assert "@param tempInKelvin the temperature, in kelvin" in committed
    final = flow.store.get(proposal.short_id)
    assert final.accepted_mechanism == "regeneration" and final.commit_sha == "abcdef1234567"
    assert kb.stats("acme", "demo")["counts"]["validated"] == 1


def test_accepting_a_regeneration_that_does_not_exist_is_refused(monkeypatch, tmp_path):
    fake = ChecksGithub()
    flow = flow_for(monkeypatch, tmp_path, fake, responder())
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    outcome = flow.decide(
        proposal.short_id, "accept", owner="acme", repo="demo", pr_number=1, mechanism="regeneration"
    )
    assert outcome.status == DecisionStatus.INVALID_EDIT and not fake.commits
    assert not flow.store.get(proposal.short_id).resolved


def test_accept_with_an_edit_is_checked_by_the_exact_rules(monkeypatch, tmp_path):
    fake = ChecksGithub()
    kb = KnowledgeBase(tmp_path / "state.sqlite3")
    flow = flow_for(monkeypatch, tmp_path, fake, responder(), knowledge=kb)
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    broken = "/**\n * Converts.\n * @param nothing x\n * @return y\n */"
    refused = flow.decide(
        proposal.short_id, "accept", owner="acme", repo="demo", pr_number=1, mechanism="edited", edited_doc=broken
    )
    assert refused.status == DecisionStatus.INVALID_EDIT and "unknown-param" in refused.message
    assert not fake.commits and not flow.store.get(proposal.short_id).resolved
    edited = (
        "/**\n * Converts kelvin to celsius (edited).\n * @param tempInKelvin the temperature\n"
        " * @return celsius\n */"
    )
    applied = flow.decide(
        proposal.short_id, "accept", owner="acme", repo="demo", pr_number=1, mechanism="edited", edited_doc=edited
    )
    assert applied.status == DecisionStatus.APPLIED
    committed = base64.b64decode(fake.commits[0]["content"]).decode()
    assert "     * Converts kelvin to celsius (edited)." in committed  # laid out like the file
    stored = flow.store.get(proposal.short_id)
    assert stored.accepted_mechanism == "edited" and "(edited)" in stored.edited_doc
    assert "(edited)" in kb.chunks("acme", "demo", ["validated"])[0].text  # the edit teaches the base


def test_edited_text_rules():
    from docdebt.core.java_parser import parse_java_source

    unit = parse_java_source(SOURCE, "T.java")[0]
    assert edited_javadoc_problems(unit, "not a javadoc")
    assert edited_javadoc_problems(unit, unit.javadoc)  # identical: nothing would change
    assert (
        normalize_edited_javadoc("/**\n* a\n  * @return b\n*/", "    ")
        == "/**\n     * a\n     * @return b\n     */"
    )


def test_slash_commands_regenerate_and_accept_regenerated():
    assert parse_slash_command("/docdebt regenerate 1a2b3c4d").decision == "regenerate"
    assert parse_slash_command("/docdebt accept 1a2b3c4d regenerated").mechanism == "regeneration"
    assert parse_slash_command("/docdebt accept 1a2b3c4d").mechanism == "replacement"
    assert parse_slash_command("/docdebt reject 1a2b3c4d false-positive").reason == "false_positive"
    assert parse_slash_command("/docdebt reject 1a2b3c4d").reason is None


def test_a_reject_command_without_reason_asks_for_one(monkeypatch, tmp_path):
    from .test_service_review_flow import comment_event

    fake = ChecksGithub()
    flow = flow_for(monkeypatch, tmp_path, fake, responder())
    flow.review_pull_request(pr_event())
    [proposal] = flow.store.pending_for_pr("acme", "demo", 1)
    text = f"/docdebt reject {proposal.short_id}"
    flow.handle_command(comment_event(text), parse_slash_command(text))
    assert "needs its reason" in fake.comments[-1] and not flow.store.get(proposal.short_id).resolved


# -- learning: plain statistics ------------------------------------------------------------------


def test_precision_by_model_and_committed_mechanism(store):
    for i in range(3):
        sid = add(store, f"A.{i}", detection_model="gpt-5.6-luna", repair_mechanism="replacement")
        store.update_proposal(sid, accepted_mechanism="edited" if i == 0 else "replacement")
        store.mark_resolved(sid, "accept")
    store.mark_resolved(add(store, "B.0", repair_mechanism="replacement"), "reject", "false_positive")
    store.mark_resolved(add(store, "C.e"), "expired")  # nobody decided: excluded
    p = precision(store.decided_proposals())
    assert (p["decisions"], p["accepted"], p["precision"]) == (4, 3, 0.75)
    assert {r["key"]: r["accepted"] for r in p["by_mechanism"]} == {"replacement": 2, "edited": 1}
    assert p["reject_reasons"] == {"false_positive": 1} and "enough" not in p


# -- documentation health and the knowledge base ------------------------------------------------------


def test_measure_counts_methods_documentation_and_debt():
    figures = measure({"src/main/java/demo/Temperature.java": SOURCE})
    assert (figures["methods"], figures["documented"], figures["debt_methods"]) == (1, 1, 1)
    assert figures["score"] == 0.0


class RepoGithub(FakeGithub):
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


def test_health_scan_indexes_the_base_and_reports_progress(monkeypatch, tmp_path):
    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    github = GithubClient("1", "k", "9", http=httpx.Client(transport=httpx.MockTransport(RepoGithub().handler)))
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    progress = []
    values = run_health_scan(github, "acme", "demo", knowledge=kb, progress=lambda d, t: progress.append((d, t)))
    assert values["score"] == 0.5 and values["commit_sha"] == "head1"
    assert kb.stats("acme", "demo")["counts"] == {"method": 1, "test": 1, "readme": 1, "validated": 0}
    assert progress[-1][0] == progress[-1][1] and len(progress) == progress[-1][1]


def test_a_push_to_the_default_branch_reindexes_the_changed_files(monkeypatch, tmp_path):
    payload = {
        "ref": "refs/heads/main",
        "after": "head2",
        "repository": {"name": "demo", "owner": {"login": "acme"}, "default_branch": "main"},
        "installation": {"id": 9},
        "commits": [
            {"added": [], "modified": ["src/main/java/demo/Ok.java"], "removed": []},
            {"added": [], "modified": [], "removed": ["src/main/java/demo/Temperature.java"]},
        ],
    }
    event = parse_push_event(payload)
    assert (event.changed, event.removed, event.after_sha) == (
        ["src/main/java/demo/Ok.java"],
        ["src/main/java/demo/Temperature.java"],
        "head2",
    )
    assert parse_push_event({**payload, "ref": "refs/heads/feature"}) is None  # other branches: ignored

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    github = GithubClient("1", "k", "9", http=httpx.Client(transport=httpx.MockTransport(RepoGithub().handler)))
    kb = KnowledgeBase(tmp_path / "kb.sqlite3")
    run_health_scan(github, "acme", "demo", knowledge=kb)
    stats = run_push_index(
        github, "acme", "demo", after_sha="head2", changed=event.changed, removed=event.removed, knowledge=kb
    )
    assert stats["commit_sha"] == "head2" and stats["counts"]["method"] == 1


# -- API routes -----------------------------------------------------------------------------------


def test_status_banner_and_getting_started(client, store, settings):
    status = client.get("/api/status").json()
    steps = {s["key"]: s["done"] for s in status["onboarding"]}
    assert steps == {
        "sign_in": True,
        "install_app": False,
        "configure_model": False,
        "index_knowledge": False,
        "first_pull_request": False,
    }
    assert status["model"]["configured"] is False and status["knowledge"]["embeddings"] == "bm25"
    store.connect_repository(owner="acme", repo="demo", installation_id="9")
    client.put("/api/llm-config", json={"model": "gpt-5.6-luna", "api_key": KEY}, headers=AUTH)
    status = client.get("/api/status").json()
    steps = {s["key"]: s["done"] for s in status["onboarding"]}
    assert steps["install_app"] and steps["configure_model"] and not steps["index_knowledge"]
    assert status["model"]["ok"] is True and status["knowledge"]["not_indexed"] == ["acme/demo"]


def test_review_queue_costs_and_repo_settings_routes(client, store):
    first, second = add(store, "A.f"), add(store, "B.g")
    store.mark_resolved(second, "accept")
    assert [p["short_id"] for p in client.get("/api/review-queue").json()] == [first]
    costs = client.get("/api/costs").json()
    assert costs["total_usd"] == 0 and costs["cap_usd"] is None
    assert client.get("/api/repos/acme/demo/settings").json()["block_on_debt"] is False
    saved = client.put("/api/repos/acme/demo/settings", json={"block_on_debt": True}, headers=AUTH).json()
    assert saved["block_on_debt"] is True
    assert client.get("/api/repos/acme/demo/threshold-suggestion").status_code == 404  # no threshold anymore


def test_api_regenerate_and_decide_with_an_edit(client, store, monkeypatch):
    from docdebt.service import app as app_module
    from docdebt.service.review_flow import DecisionOutcome

    seen = {}

    def fake_decide(settings, store_, request, decision, installation_id, **kwargs):
        seen.update(decision=decision, **kwargs)
        return DecisionOutcome(DecisionStatus.INVALID_EDIT, "unknown-param: phantom")

    monkeypatch.setattr(app_module, "_decide_from_dashboard", fake_decide)
    sid = add(store)
    no_reason = client.post(f"/api/proposals/{sid}/decide", json={"decision": "reject"}, headers=AUTH)
    assert no_reason.status_code == 422
    no_text = client.post(
        f"/api/proposals/{sid}/decide", json={"decision": "accept", "mechanism": "edited"}, headers=AUTH
    )
    assert no_text.status_code == 422
    edit = client.post(
        f"/api/proposals/{sid}/decide",
        json={"decision": "accept", "mechanism": "edited", "edited_doc": "/** x */"},
        headers=AUTH,
    )
    assert edit.status_code == 422 and "unknown-param" in edit.json()["detail"]
    assert seen["mechanism"] == "edited" and seen["edited_doc"] == "/** x */"
    detail = client.get(f"/api/proposals/{sid}").json()
    assert detail["regeneration"] == {"status": None, "doc": None, "screened": None}
    assert client.post("/api/proposals/nothing00/regenerate", headers=AUTH).status_code == 404
