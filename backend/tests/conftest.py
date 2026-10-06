"""Shared test helpers: a scripted LangChain chat model and sample Java code."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from docdebt.llm.config import ModelSpec

Responder = Callable[[str, int], tuple[str, str]]  # (prompt, max_tokens) -> (text, finish_reason)


class ScriptedChatModel(BaseChatModel):
    """Fake chat model whose answers come from a Python function."""

    responder: Any
    max_tokens: int = 0
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        prompt = str(messages[-1].content)
        self.calls.append(prompt)
        result = self.responder(prompt, self.max_tokens)
        if isinstance(result, Exception):
            raise result
        text, reason = result
        message = AIMessage(
            content=text,
            response_metadata={"done_reason": reason},
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_factory(responder: Responder, calls: list | None = None):
    calls = calls if calls is not None else []

    def factory(spec: ModelSpec, max_tokens: int, temperature: float | None) -> BaseChatModel:
        return ScriptedChatModel(responder=responder, max_tokens=max_tokens, calls=calls)

    return factory


@pytest.fixture
def local_spec() -> ModelSpec:
    return ModelSpec(
        id="qwen2.5-coder:7b",
        provider="ollama",
        model="qwen2.5-coder:7b",
        max_tokens={"default": 60, "cot": 1200, "explain": 600},
        max_retries=3,
    )


@pytest.fixture
def cloud_spec() -> ModelSpec:
    return ModelSpec(
        id="gpt-5.6",
        provider="openai",
        model="gpt-5.6",
        max_tokens={"default": 500},
        max_retries=3,
        usd_per_1k_input=1.0,
        usd_per_1k_output=2.0,
    )


KELVIN_SOURCE = """package demo;

public class Temperature {

    /**
     * Converts kelvin to celsius.
     * @param tempInCelsius temperature in kelvin
     * @return temperature in celsius
     */
    public double toCelsius(double tempInKelvin) {
        return tempInKelvin - 273.15;
    }

    /**
     * Returns the maximum of the two values.
     * @param a first value
     * @param b second value
     * @return the smaller of a and b
     */
    public int pick(int a, int b) {
        return Math.min(a, b);
    }

    /**
     * Reads a file.
     * @param path the path
     * @return the content
     */
    public String read(String path) throws java.io.IOException {
        return helper(path);
    }

    private String helper(String p) { return p; }
}
"""
