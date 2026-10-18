import base64
import json

import httpx

from docdebt.core.auditor import Auditor
from docdebt.graph.build import GraphApp
from docdebt.graph.nodes import GraphDeps
from docdebt.llm.client import LLMClient
from docdebt.repair.registry import RepairRegistry
from docdebt.repair.repairer import RepairConfig, Repairer
from docdebt.service.commands import parse_slash_command
from docdebt.service.github_client import GithubClient
from docdebt.service.review_flow import ReviewFlow, extract_touched_units
from docdebt.service.store import ReviewStore
from docdebt.service.webhooks import CommentEvent, PullRequestEvent

from .conftest import make_factory

SOURCE = """package demo;

public class Temperature {

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) {
        return tempInKelvin - 273.15;
    }
}
"""

DIFF = """diff --git a/src/Temperature.java b/src/Temperature.java
--- a/src/Temperature.java
+++ b/src/Temperature.java
@@ -9,1 +9,1 @@
-    public double toCelsius(double tempInKelvin) {
+    public double toCelsius(double tempInKelvin) {
"""

FIXED_JAVADOC = (
    "/**\n * Converts kelvin to celsius.\n"
    " * @param tempInKelvin temperature in kelvin\n"
    " * @return temperature in celsius\n */"
)


class FakeGithub:
    """Records every call a real GithubClient would make, without touching the network.

    ``content_at_accept``, when given, is returned only from the SECOND GET on
    ``/contents/`` onward, so a test can model the PR's file changing on GitHub
    between the review being posted (first fetch) and the accept being handled
    (second fetch) -- the exact scenario the "file changed" safety check exists
    for. Without it, every fetch returns ``file_content``.
    """

    def __init__(
        self,
        file_content: str = SOURCE,
        put_status: int = 200,
        content_at_accept: str | None = None,
        open_prs: tuple[int, ...] = (1,),
        failing_prs: tuple[int, ...] = (),
    ):
        self.open_prs = open_prs
        self.failing_prs = failing_prs
        self.comments: list[str] = []
        self.commits: list[dict] = []
        self.file_content = file_content
        self.content_at_accept = content_at_accept
        self.put_status = put_status
        self.file_sha = "sha-v1"
        self._content_gets = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        if path.endswith("/pulls") and method == "GET":
            return httpx.Response(
                200, json=[{"number": n, "head": {"ref": f"branch-{n}", "sha": f"sha-{n}"}} for n in self.open_prs]
            )
        if "/pulls/" in path and request.headers.get("accept", "").endswith("v3.diff"):
            if int(path.rsplit("/", 1)[1]) in self.failing_prs:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, text=DIFF)
        if "/pulls/" in path and method == "GET":
            return httpx.Response(200, json={"number": 1, "head": {"ref": "feature", "sha": "sha1"}})
        if "/contents/" in path and method == "GET":
            self._content_gets += 1
            content = self.file_content
            if self._content_gets > 1 and self.content_at_accept is not None:
                content = self.content_at_accept
            return httpx.Response(
                200, json={"content": base64.b64encode(content.encode()).decode(), "sha": self.file_sha}
            )
        if "/contents/" in path and method == "PUT":
            if self.put_status != 200:
                return httpx.Response(self.put_status, json={"message": "sha mismatch"})
            body = json.loads(request.content)
            self.commits.append(body)
            return httpx.Response(200, json={"commit": {"sha": "abcdef1234567"}})
        if path.endswith("/comments") and method == "POST":
            body = json.loads(request.content)
            self.comments.append(body["body"])
            return httpx.Response(201, json={"id": len(self.comments)})
        if "/reactions" in path:
            return httpx.Response(200, json={"id": 1})
        raise AssertionError(f"unexpected call: {method} {path}")


def make_flow(monkeypatch, fake, tmp_path, responder):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    http = httpx.Client(transport=httpx.MockTransport(fake.handler))
    github = GithubClient("app-1", "irrelevant", "install-1", http=http)

    from docdebt.llm.config import ModelSpec

    spec = ModelSpec(
        id="qwen2.5-coder:7b",
        provider="ollama",
        model="qwen2.5-coder:7b",
        max_tokens={"default": 60, "repair": 1400},
        max_retries=3,
    )
    repairer = Repairer(
        LLMClient(spec, factory=make_factory(responder), sleep=lambda s: None),
        RepairRegistry(),
        config=RepairConfig(),
    )
    deps = GraphDeps(auditor=Auditor(), router=None, repairer=repairer)
    graph = GraphApp(deps, InMemorySaver())
    store = ReviewStore(tmp_path / "state.sqlite3")
    return ReviewFlow(github, graph, store, spec=spec)


def structural_fix_responder(prompt, m):
    if "CURRENT DOCUMENTATION:" in prompt:
        return FIXED_JAVADOC, "stop"
    return "NONE", "stop"


def pr_event(**overrides):
    defaults = dict(
        action="opened",
        owner="acme",
        repo="demo",
        number=1,
        head_sha="sha1",
        head_ref="feature",
        installation_id="install-1",
    )
    return PullRequestEvent(**{**defaults, **overrides})


def comment_event(body, **overrides):
    defaults = dict(
        action="created",
        owner="acme",
        repo="demo",
        issue_number=1,
        comment_id=42,
        body=body,
        installation_id="install-1",
        is_pull_request=True,
        author_type="User",
    )
    return CommentEvent(**{**defaults, **overrides})


def test_extract_touched_units_from_api_fetched_content():
    from docdebt.core.extractor import Extractor

    units = extract_touched_units(DIFF, {"src/Temperature.java": SOURCE}, Extractor())
    assert [u.name for u in units] == ["toCelsius"]


def test_review_pull_request_posts_one_proposal_with_a_short_id(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    outcome = flow.review_pull_request(pr_event())
    assert outcome.proposed == 1
    assert len(fake.comments) == 1
    assert "/docdebt accept " in fake.comments[0] and "tempInCelsius" in fake.comments[0]
    # our own proposal text DOES contain a matching "/docdebt accept <id>" pattern --
    # it spells out the instruction for a human to type. This is exactly why
    # CommentEvent.from_bot exists and is enforced at the webhook dispatch layer
    # (see test_service_app.py's bot-comment test), not here.


def test_accept_commits_the_fix_and_only_the_fix(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())

    proposal = fake.comments[0]
    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", proposal).group(1)
    flow.handle_command(
        comment_event(f"/docdebt accept {short_id}"), parse_slash_command(f"/docdebt accept {short_id}")
    )

    assert len(fake.commits) == 1
    new_content = base64.b64decode(fake.commits[0]["content"]).decode()
    assert "@param tempInKelvin" in new_content and "@param tempInCelsius" not in new_content
    assert new_content.count("\n") == SOURCE.count("\n")  # no stray line added or removed
    assert fake.commits[0]["sha"] == fake.file_sha and fake.commits[0]["branch"] == "feature"
    assert "✅" in fake.comments[-1]


def test_reject_makes_no_commit(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)
    flow.handle_command(
        comment_event(f"/docdebt reject {short_id} other"),
        parse_slash_command(f"/docdebt reject {short_id} other"),
    )
    assert not fake.commits
    assert "rejected" in fake.comments[-1]


def test_unknown_short_id_gets_a_friendly_reply_no_crash(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.handle_command(comment_event("/docdebt accept deadbeef"), parse_slash_command("/docdebt accept deadbeef"))
    assert not fake.commits
    assert "No pending review" in fake.comments[-1]


def test_double_accept_is_a_no_op_the_second_time(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)
    cmd = parse_slash_command(f"/docdebt accept {short_id}")
    flow.handle_command(comment_event(f"/docdebt accept {short_id}"), cmd)
    flow.handle_command(comment_event(f"/docdebt accept {short_id}"), cmd)
    assert len(fake.commits) == 1  # the second reply must not re-commit


def test_accept_after_file_changed_on_github_is_reported_not_crashed(monkeypatch, tmp_path):
    fake = FakeGithub(content_at_accept=SOURCE.replace("Converts kelvin", "Something else entirely"))
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)
    flow.handle_command(
        comment_event(f"/docdebt accept {short_id}"), parse_slash_command(f"/docdebt accept {short_id}")
    )
    assert not fake.commits
    assert "file changed" in fake.comments[-1] or "push again" in fake.comments[-1]


def test_accept_when_github_rejects_the_commit_is_reported_not_crashed(monkeypatch, tmp_path):
    fake = FakeGithub(put_status=422)
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)
    flow.handle_command(
        comment_event(f"/docdebt accept {short_id}"), parse_slash_command(f"/docdebt accept {short_id}")
    )
    assert not fake.commits
    assert "422" in fake.comments[-1]


def test_clean_method_produces_no_comment(monkeypatch, tmp_path):
    clean_source = SOURCE.replace(
        "@param tempInCelsius temperature in kelvin", "@param tempInKelvin temperature in kelvin"
    )
    fake = FakeGithub(file_content=clean_source)
    flow = make_flow(monkeypatch, fake, tmp_path, lambda p, m: ("COMPLETE", "stop"))
    outcome = flow.review_pull_request(pr_event())
    assert outcome.proposed == 0 and outcome.skipped_no_debt == 1
    assert not fake.comments


# -- dashboard data: usage logging and installations tracking ----------------


def test_review_pull_request_logs_llm_usage(monkeypatch, tmp_path):
    from docdebt.llm.client import UsageTracker

    fake = FakeGithub()

    def make_flow_with_shared_tracker():
        from langgraph.checkpoint.memory import InMemorySaver

        monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
        http = httpx.Client(transport=httpx.MockTransport(fake.handler))
        github = GithubClient("app-1", "irrelevant", "install-1", http=http)

        from docdebt.llm.config import ModelSpec

        spec = ModelSpec(
            id="qwen2.5-coder:7b",
            provider="ollama",
            model="qwen2.5-coder:7b",
            max_tokens={"default": 60, "repair": 1400},
            max_retries=3,
        )
        tracker = UsageTracker()
        repairer = Repairer(
            LLMClient(spec, factory=make_factory(structural_fix_responder), sleep=lambda s: None, tracker=tracker),
            RepairRegistry(),
            config=RepairConfig(),
        )
        deps = GraphDeps(auditor=Auditor(), router=None, repairer=repairer)
        graph = GraphApp(deps, InMemorySaver())
        store = ReviewStore(tmp_path / "state.sqlite3")
        return ReviewFlow(github, graph, store, tracker=tracker), store

    flow, store = make_flow_with_shared_tracker()
    flow.review_pull_request(pr_event())

    summary = store.summary()
    assert summary["llm_calls"] > 0 and summary["cost_usd"] >= 0
    series = store.usage_series(days=1)
    assert series[-1]["calls"] == summary["llm_calls"]


def test_repo_breakdown_and_recent_proposals(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())

    breakdown = flow.store.repo_breakdown()
    assert breakdown == [
        {
            "owner": "acme",
            "repo": "demo",
            "proposals": 1,
            "pending": 1,
            "accepted": 0,
            "rejected": 0,
            "expired": 0,
            "unknown": 0,
            "cost_usd": 0.0,
        }
    ]

    recent = flow.store.recent_proposals(limit=5)
    assert len(recent) == 1 and recent[0].method_label == "Temperature.toCelsius"

    import re

    short_id = re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)
    flow.handle_command(
        comment_event(f"/docdebt accept {short_id}"), parse_slash_command(f"/docdebt accept {short_id}")
    )
    breakdown = flow.store.repo_breakdown()
    assert breakdown[0]["accepted"] == 1 and breakdown[0]["pending"] == 0


# -- decisions from the dashboard ---------------------------------------------


def _short_id(fake):
    import re

    return re.search(r"/docdebt accept ([0-9a-f]+)", fake.comments[0]).group(1)


def test_review_stores_the_installation_id_for_dashboard_decisions(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event(installation_id="install-9"))
    assert flow.store.get(_short_id(fake)).installation_id == "install-9"


def test_dashboard_accept_commits_like_a_comment_and_leaves_a_trace_on_the_pr(monkeypatch, tmp_path):
    from docdebt.service.review_flow import DecisionStatus

    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())

    outcome = flow.decide(_short_id(fake), "accept", owner="acme", repo="demo", pr_number=1, via="the dashboard")

    assert outcome.status == DecisionStatus.APPLIED and outcome.commit_sha == "abcdef1"
    assert len(fake.commits) == 1
    assert "@param tempInKelvin" in base64.b64decode(fake.commits[0]["content"]).decode()
    assert "✅" in fake.comments[-1] and "decided from the dashboard" in fake.comments[-1]
    assert flow.store.get(_short_id(fake)).decision == "accept"


def test_dashboard_reject_makes_no_commit(monkeypatch, tmp_path):
    from docdebt.service.review_flow import DecisionStatus

    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())

    outcome = flow.decide(_short_id(fake), "reject", owner="acme", repo="demo", pr_number=1, via="the dashboard")
    assert outcome.status == DecisionStatus.REJECTED
    assert not fake.commits
    assert "rejected (decided from the dashboard)" in fake.comments[-1]


def test_dashboard_then_comment_on_the_same_proposal_commits_once(monkeypatch, tmp_path):
    from docdebt.service.review_flow import DecisionStatus

    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    short_id = _short_id(fake)

    flow.decide(short_id, "accept", owner="acme", repo="demo", pr_number=1, via="the dashboard")
    cmd = parse_slash_command(f"/docdebt accept {short_id}")
    outcome = flow.handle_command(comment_event(f"/docdebt accept {short_id}"), cmd)
    assert outcome.status == DecisionStatus.NOT_FOUND
    assert len(fake.commits) == 1


def test_dashboard_accept_when_github_rejects_the_commit_reports_not_applied(monkeypatch, tmp_path):
    from docdebt.service.review_flow import DecisionStatus

    fake = FakeGithub(put_status=422)
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())

    outcome = flow.decide(_short_id(fake), "accept", owner="acme", repo="demo", pr_number=1, via="the dashboard")
    assert outcome.status == DecisionStatus.NOT_APPLIED and "422" in outcome.message
    assert not fake.commits


def test_decision_on_a_proposal_whose_graph_run_is_gone_expires_it_cleanly(monkeypatch, tmp_path):
    """A proposal row whose paused graph thread no longer exists (graph database
    reset or moved, or a row inserted by hand) must not restart the graph from
    an empty state -- that crashed with KeyError: 'unit' -- but expire cleanly."""
    from docdebt.service.review_flow import DecisionStatus

    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    short_id = flow.store.create(
        thread_id="acme/demo#4:no-such-thread",
        owner="acme",
        repo="demo",
        pr_number=4,
        file_path="src/Temperature.java",
        method_label="Temperature.toCelsius",
        head_sha="sha1",
        installation_id="install-1",
    )

    outcome = flow.decide(short_id, "accept", owner="acme", repo="demo", pr_number=4, via="the dashboard")

    assert outcome.status == DecisionStatus.EXPIRED
    assert not fake.commits
    assert "no longer available" in fake.comments[-1]
    assert flow.store.get(short_id).resolved and flow.store.get(short_id).decision == "expired"


# -- re-reviews and on-demand scans --------------------------------------------


def test_reviewing_the_same_pr_again_does_not_repost_a_pending_proposal(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    first = flow.review_pull_request(pr_event())
    second = flow.review_pull_request(pr_event(action="synchronize"))
    assert (first.proposed, second.proposed, second.already_proposed) == (1, 0, 1)
    assert len(fake.comments) == 1
    assert len(flow.store.recent_proposals()) == 1


def test_scan_reviews_every_open_pr_and_survives_a_failing_one(monkeypatch, tmp_path):
    fake = FakeGithub(open_prs=(1, 2, 3), failing_prs=(2,))
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    outcome = flow.scan_repository("acme", "demo", "install-1")
    assert (outcome.prs, outcome.proposed, outcome.failed_prs) == (3, 2, 1)
    assert {r.pr_number for r in flow.store.recent_proposals()} == {1, 3}

    again = flow.scan_repository("acme", "demo", "install-1")  # nothing new since the first scan
    assert (again.proposed, again.already_proposed) == (0, 2)
    assert len(fake.comments) == 2


def test_a_method_with_debt_but_no_verified_fix_is_logged_with_the_reason(monkeypatch, tmp_path, caplog):
    import logging

    fake = FakeGithub()
    # The model keeps the phantom @param: the Verifier rejects every attempt.
    flow = make_flow(
        monkeypatch, fake, tmp_path, lambda prompt, m: (SOURCE.split("    /**")[1].split("*/")[0], "stop")
    )
    with caplog.at_level(logging.INFO, logger="docdebt.service"):
        outcome = flow.review_pull_request(pr_event())
    assert outcome.skipped_not_proposed == 1 and not fake.comments
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("no fix proposed"))
    assert "Temperature.toCelsius" in line and "src/Temperature.java" in line and "attempt" in line


def test_a_decided_proposal_is_counted_as_decided_not_as_a_failed_repair(monkeypatch, tmp_path):
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    flow.review_pull_request(pr_event())
    flow.decide(_short_id(fake), "reject", owner="acme", repo="demo", pr_number=1)

    again = flow.review_pull_request(pr_event(action="synchronize"))  # same content: the rejection stands
    assert (again.already_decided, again.skipped_not_proposed, again.proposed) == (1, 0, 0)
    assert sum("Documentation debt" in c for c in fake.comments) == 1  # not proposed a second time


# -- proposal details (what the dashboard shows before a decision) ----------------


def test_proposal_details_show_the_debt_the_method_and_the_proposed_fix(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from docdebt.graph.build import sqlite_checkpointer
    from docdebt.service.app import create_app
    from docdebt.service.config import ServiceSettings

    settings = ServiceSettings(
        github_app_id="1",
        github_app_slug="app",
        github_private_key="k",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
    )
    fake = FakeGithub()
    flow = make_flow(monkeypatch, fake, tmp_path, structural_fix_responder)
    with sqlite_checkpointer(settings.graph_checkpoint_path) as saver:  # the real, on-disk checkpointer
        flow.graph = GraphApp(flow.graph.deps, saver)
        flow.review_pull_request(pr_event())
    short_id = _short_id(fake)

    client = TestClient(create_app(settings))
    body = client.get(f"/api/proposals/{short_id}").json()
    details = body["details"]

    assert body["method_label"] == "Temperature.toCelsius" and body["decision"] is None
    assert details["method"]["signature"] == "double toCelsius(double tempInKelvin)"
    assert "return tempInKelvin - 273.15;" in details["method"]["code"]
    assert (
        "    /**\n" in details["method"]["source"]
        and "\n    public double toCelsius" in details["method"]["source"]
    )
    assert {f["rule"] for f in details["debt"]} == {"missing-param", "unknown-param"}
    assert "-     * @param tempInCelsius temperature in kelvin" in details["diff"]
    assert "+     * @param tempInKelvin temperature in kelvin" in details["diff"]
    assert {(c["field"], c["action"]) for c in details["repair"]["changes"]} == {
        ("@param tempInCelsius", "remove"),
        ("@param tempInKelvin", "add"),
    }
    assert details["repair"]["status"] == "verified" and details["repair"]["attempts"] == 1

    assert client.get("/api/proposals/deadbeef").status_code == 404
    assert client.get("/api/proposals/search").status_code == 200  # not taken for a proposal id


def test_proposal_details_are_none_when_the_run_is_gone(tmp_path):
    from fastapi.testclient import TestClient

    from docdebt.service.app import create_app
    from docdebt.service.config import ServiceSettings

    settings = ServiceSettings(
        github_app_id="1",
        github_app_slug="app",
        github_private_key="k",
        github_webhook_secret="s",
        state_db_path=tmp_path / "state.sqlite3",
        graph_checkpoint_path=tmp_path / "graph.sqlite3",
    )
    short_id = ReviewStore(settings.state_db_path).create(
        thread_id="no-such-thread",
        owner="acme",
        repo="demo",
        pr_number=1,
        file_path="X.java",
        method_label="X.f",
        head_sha="s",
    )
    body = TestClient(create_app(settings)).get(f"/api/proposals/{short_id}").json()
    assert body["short_id"] == short_id and body["details"] is None
