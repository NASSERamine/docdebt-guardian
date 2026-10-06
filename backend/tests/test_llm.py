import warnings

import pytest

from docdebt.llm.client import LLMClient, LLMError, UsageTracker, clean_model_text
from docdebt.llm.parsing import parse_binary_verdict, parse_claim_citations
from docdebt.llm.prompts.detection_wordings import GeneralExample
from docdebt.llm.prompts.fewshot import FewShotSet
from docdebt.llm.prompts.registry import PromptRegistry, append_context

from .conftest import make_factory


def test_verdict_parsing_last_word_wins():
    assert parse_binary_verdict("INCOMPLETE") is True
    assert parse_binary_verdict("complete") is False
    assert parse_binary_verdict("Looks INCOMPLETE at first...\nFinal Answer: COMPLETE") is False
    assert parse_binary_verdict("I cannot tell") is None
    assert parse_binary_verdict(None) is None


def test_claim_parsing_with_braces_and_none():
    raw = (
        "CLAIM: Returns the {@code maximum} of the two values.\n"
        "CODE: return Math.min(a, b);\nREASON: min not max\nFIELD: description\n---\n"
    )
    claims = parse_claim_citations(raw)
    assert len(claims) == 1 and claims[0].code == "return Math.min(a, b);"
    assert parse_claim_citations("NONE") == []
    assert parse_claim_citations("garbage output") is None


def test_client_retries_truncation_with_bigger_budget(local_spec):
    seen = []

    def responder(prompt, max_tokens):
        seen.append(max_tokens)
        return ("partial", "length") if max_tokens < 240 else ("COMPLETE", "stop")

    tracker = UsageTracker()
    client = LLMClient(local_spec, factory=make_factory(responder), tracker=tracker, sleep=lambda s: None)
    response = client.invoke("p")
    assert seen == [60, 120, 240]
    assert response.text == "COMPLETE" and not response.truncated and response.attempts == 3
    assert tracker.calls == 3


def test_client_returns_truncated_when_cap_reached(local_spec):
    spec = local_spec.model_copy(update={"max_tokens_cap": 60})
    client = LLMClient(spec, factory=make_factory(lambda p, m: ("par", "length")), sleep=lambda s: None)
    response = client.invoke("p")
    assert response.truncated and response.attempts == 1


def test_client_drops_unsupported_temperature(cloud_spec, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    temperatures = []

    def factory(spec, max_tokens, temperature):
        temperatures.append(temperature)
        if temperature is not None:
            return make_factory(lambda p, m: ValueError("temperature is not supported"))(
                spec, max_tokens, temperature
            )
        return make_factory(lambda p, m: ("COMPLETE", "stop"))(spec, max_tokens, temperature)

    client = LLMClient(cloud_spec, factory=factory, sleep=lambda s: None)
    assert client.invoke("p").text == "COMPLETE"
    assert temperatures == [0.0, None]
    assert client.tracker.cost_usd == pytest.approx((10 * 1.0 + 2 * 2.0) / 1000)


def test_client_raises_after_errors(local_spec):
    client = LLMClient(
        local_spec, factory=make_factory(lambda p, m: ConnectionError("down")), sleep=lambda s: None
    )
    with pytest.raises(LLMError):
        client.invoke("p")


def test_think_blocks_removed():
    assert clean_model_text("<think>hmm</think>\nCOMPLETE") == "COMPLETE"


def test_registry_locked_wordings_and_versions():
    registry = PromptRegistry()
    p = registry.build("qwen2.5-coder:7b", "cot", "int f(){return 1;}", "/** d */")
    assert p.wording_index == 2 and p.version.startswith("detect/cot/w2/")
    assert "Final Answer" in p.text
    with pytest.raises(KeyError):
        registry.build("unknown-model", "zeroshot", "c", "d")
    with pytest.raises(ValueError):
        registry.build("qwen2.5-coder:7b", "oneshot", "c", "d")  # needs few-shot file
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry.build("qwen2.5-coder:7b", "persona", "c", "d")
    assert any("collapsed" in str(w.message) for w in caught)


def test_context_closing_matches_pattern():
    assert append_context("P", "", "cot") == "P"
    assert append_context("P", "ctx", "cot").endswith("Final Answer: INCOMPLETE")
    assert append_context("P", "ctx", "zeroshot").endswith("COMPLETE or INCOMPLETE.")


def test_fewshot_order():
    examples = [GeneralExample(f"code{i}", f"doc{i}", "COMPLETE") for i in range(6)]
    fs = FewShotSet(examples)
    assert [e.method_code for e in fs.for_pattern("oneshot")] == ["code0", "code4"]
    assert [e.method_code for e in fs.for_pattern("threeshot")] == ["code0", "code2", "code4"]
    with pytest.raises(ValueError):
        FewShotSet(examples[:5])
    registry = PromptRegistry(fewshot=fs)
    assert "code4" in registry.build("qwen2.5-coder:7b", "oneshot", "c", "d").text
