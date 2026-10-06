import base64
import time

import httpx
import pytest

from docdebt.service.github_client import GithubApiError, GithubClient

FAKE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF0pxTz3xIcIEfESmB5DGfaOF/lm0
p9c1LtBS2ByfW7Ep9nCynqZFm2ZTYTx3wJ2Uahd7ajl67NCd8FmyGbn45neSbmSD
UBohNCA8lo22z7CqDWbnPX0PdyoQR6qDPUZGpjmZ8WIcxx8+3xn3UByGKF2IVXKo
gNCUUjyeCBcCVFCH3Vj/lXAJ/oFPWStpjqzeUmpNPFO8oa2Ivh/O2XV3Vf1S8jhh
sKZWMr7HFhg8ZwoJmxTeS3lTXlq9jP1G4qBjLrqf4PVBJ9nYCzHzOm4E8m/N4jlv
xUgt5j+8nA0Ncr9y5aeAt+McQ9UgmvUJx87HNwIDAQABAoIBAAXo6L4qXFvxUvyR
b4NfycygQFEr3nO24Q3SxSGCT/vwmVCkGF2n6IqIRIvFm/HmuVfANNvzBihaR0T2
example-placeholder-not-a-real-key-only-used-to-satisfy-pyjwt-encode-shape
-----END RSA PRIVATE KEY-----"""


@pytest.fixture
def client(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "ghs_fake", "expires_at": "2099-01-01T00:00:00Z"})
        if "/contents/" in request.url.path and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "content": base64.b64encode(b"public class A {}").decode(),
                    "sha": "abc123",
                },
            )
        if "/contents/" in request.url.path and request.method == "PUT":
            return httpx.Response(200, json={"commit": {"sha": "def456"}})
        if request.url.path.endswith("/comments") and request.method == "POST":
            return httpx.Response(201, json={"id": 999, "body": "ok"})
        if "/pulls/" in request.url.path and request.headers.get("accept", "").endswith("v3.diff"):
            return httpx.Response(200, text="--- a/X.java\n+++ b/X.java\n")
        if "/pulls/" in request.url.path:
            return httpx.Response(200, json={"number": 1, "head": {"ref": "feature", "sha": "sha1"}})
        return httpx.Response(404, json={"message": "not found"})

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt.token")
    gh = GithubClient("app-1", "irrelevant-with-patched-jwt", "install-1", http=http)
    gh._calls = calls  # type: ignore[attr-defined]
    return gh


def test_token_is_fetched_once_and_cached(client):
    client._installation_token()
    client._installation_token()
    token_calls = [c for c in client._calls if c.url.path.endswith("/access_tokens")]
    assert len(token_calls) == 1


def test_token_is_refreshed_once_expired(client):
    client._installation_token()
    client._cached.expires_at = time.time() - 1  # force expiry
    client._installation_token()
    token_calls = [c for c in client._calls if c.url.path.endswith("/access_tokens")]
    assert len(token_calls) == 2


def test_get_file_content_decodes_base64(client):
    content, sha = client.get_file_content("o", "r", "X.java", "main")
    assert content == "public class A {}" and sha == "abc123"


def test_update_file_content_sends_base64_and_sha(client):
    client.update_file_content("o", "r", "X.java", content="new", sha="abc123", branch="feature", message="m")
    put = next(c for c in client._calls if c.method == "PUT")
    import json

    body = json.loads(put.content)
    assert base64.b64decode(body["content"]).decode() == "new"
    assert body["sha"] == "abc123" and body["branch"] == "feature"


def test_create_issue_comment(client):
    result = client.create_issue_comment("o", "r", 7, "hello")
    assert result["id"] == 999


def test_pull_request_diff_uses_diff_media_type(client):
    diff = client.get_pull_request_diff("o", "r", 1)
    assert "--- a/X.java" in diff


def test_api_error_raised_with_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        return httpx.Response(422, json={"message": "nope"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt.token")
    gh = GithubClient("a", "k", "i", http=http)
    with pytest.raises(GithubApiError) as exc:
        gh.get_pull_request("o", "r", 1)
    assert exc.value.status_code == 422


def test_list_open_pull_requests_follows_pages(monkeypatch):
    pages = {1: [{"number": n} for n in range(100)], 2: [{"number": 100}]}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "t", "expires_at": "2099-01-01T00:00:00Z"})
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=pages[int(request.url.params["page"])])

    monkeypatch.setattr("jwt.encode", lambda payload, key, algorithm: "fake.jwt")
    client = GithubClient("1", "k", "2", http=httpx.Client(transport=httpx.MockTransport(handler)))
    pulls = client.list_open_pull_requests("acme", "demo")
    assert len(pulls) == 101
    assert [p["page"] for p in seen] == ["1", "2"] and seen[0]["state"] == "open"
