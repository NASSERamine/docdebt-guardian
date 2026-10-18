"""Prompt registry: which validated wording each model uses, and prompt versions.

Every prompt sent to a model carries a version string such as
``detect/zeroshot/w0/3fa2c1d9``: pattern, wording index and a hash of the
wording's source code. The version is stored with each verdict, so any
result can be traced back to the exact prompt text that produced it, and
the evaluation gate can tell when a prompt changed.

The locked indices below come from the research wording screening
(detection task, 30 paired items per model). They are 0-based indices
into ``detection_wordings.WORDING_CANDIDATES_GENERAL[pattern]``.
"""

from __future__ import annotations

import hashlib
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from docdebt.llm.prompts.detection_wordings import WORDING_CANDIDATES_GENERAL
from docdebt.llm.prompts.fewshot import FewShotSet

PATTERNS = ("zeroshot", "persona", "oneshot", "threeshot", "cot")
FEWSHOT_PATTERNS = frozenset({"oneshot", "threeshot"})

LOCKED_DETECTION_WORDINGS: dict[str, dict[str, int]] = {
    "qwen2.5-coder:7b": {"zeroshot": 0, "persona": 0, "oneshot": 0, "threeshot": 2, "cot": 2},
    "codegemma:7b": {"zeroshot": 0, "persona": 1, "oneshot": 2, "threeshot": 0, "cot": 0},
    "deepseek-coder:6.7b-instruct": {
        "zeroshot": 2,
        "persona": 0,
        "oneshot": 1,
        "threeshot": 0,
        "cot": 2,
    },
    "gpt-5.6": {"zeroshot": 0, "persona": 1, "oneshot": 0, "threeshot": 1, "cot": 0},
    "claude-sonnet-5": {"zeroshot": 0, "persona": 0, "oneshot": 1, "threeshot": 1, "cot": 2},
}

# Models that reuse the wordings screened for a sibling model of the same
# family, without their own screening. Their verdicts say so (``screened_on``).
SCREENED_ON: dict[str, str] = {"gpt-5.6-luna": "gpt-5.6"}
for _model, _sibling in SCREENED_ON.items():
    LOCKED_DETECTION_WORDINGS.setdefault(_model, dict(LOCKED_DETECTION_WORDINGS[_sibling]))

# Models offered by the application whose screened indices are not in this
# file yet: they use the first zero-shot wording, and their verdicts say the
# wording was not screened (``is_screened``). Replace with the research
# screening results when available.
UNSCREENED_DETECTION_WORDINGS: dict[str, dict[str, int]] = {"gemini-3.1-pro": {"zeroshot": 0}}
for _model, _indices in UNSCREENED_DETECTION_WORDINGS.items():
    LOCKED_DETECTION_WORDINGS.setdefault(_model, dict(_indices))


def is_screened(model_id: str) -> bool:
    """Whether ``model_id`` detects with a wording screened for it (or its family)."""
    return model_id not in UNSCREENED_DETECTION_WORDINGS and model_id in LOCKED_DETECTION_WORDINGS


# (model, pattern) pairs whose every screened wording collapsed to a constant
# answer, or that collapsed at full run. Using them triggers a warning.
COLLAPSED: frozenset[tuple[str, str]] = frozenset(
    {
        ("qwen2.5-coder:7b", "persona"),
        ("deepseek-coder:6.7b-instruct", "zeroshot"),
    }
)

_CONTEXT_HEADER = (
    "--- ADDITIONAL PROJECT CONTEXT ---\n"
    "The following is supplementary context from the same repository. Use it only to "
    "help you judge the method's documentation -- base your verdict primarily on the "
    "method's own code and documentation above.\n\n"
)
_CONTEXT_FOOTER = "\n\n--- END OF CONTEXT ---\n\n"
_CLOSING_ONE_WORD = (
    "Considering everything above, respond with exactly one word on its own line: COMPLETE or INCOMPLETE."
)
# Research issue fixed here: the research context builder always closed with the
# one-word instruction, which contradicts chain-of-thought wordings and can
# suppress their reasoning. CoT prompts keep their own answer format.
_CLOSING_COT = (
    "Considering everything above, reason as instructed, then on its own final line "
    "respond with exactly:\nFinal Answer: COMPLETE\nor\nFinal Answer: INCOMPLETE"
)


@dataclass(frozen=True)
class BuiltPrompt:
    """A prompt ready to send, with its traceability metadata."""

    text: str
    pattern: str
    wording_index: int
    version: str
    has_context: bool


def wording_version(pattern: str, index: int, fn: Callable[..., str]) -> str:
    digest = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:8]
    return f"detect/{pattern}/w{index}/{digest}"


def append_context(prompt: str, context: str | None, pattern: str) -> str:
    """Append retrieved context after the pattern prompt, with a closing
    instruction that matches the pattern's answer format.

    An empty context leaves the prompt unchanged.
    """
    if not context or not context.strip():
        return prompt
    closing = _CLOSING_COT if pattern == "cot" else _CLOSING_ONE_WORD
    return f"{prompt}\n\n{_CONTEXT_HEADER}{context.strip()}{_CONTEXT_FOOTER}{closing}"


class PromptRegistry:
    """Build detection prompts with the locked wording of each model."""

    def __init__(
        self,
        fewshot: FewShotSet | None = None,
        overrides: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.fewshot = fewshot
        self._indices = {m: dict(p) for m, p in LOCKED_DETECTION_WORDINGS.items()}
        for model_id, patterns in (overrides or {}).items():
            self._indices.setdefault(model_id, {}).update(patterns)

    def wording_index(self, model_id: str, pattern: str) -> int:
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown pattern '{pattern}'. Valid patterns: {PATTERNS}.")
        if model_id not in self._indices:
            raise KeyError(
                f"No locked detection wordings for model '{model_id}'. Run the wording "
                f"screening for it and add its indices to the configuration."
            )
        if pattern not in self._indices[model_id]:
            raise KeyError(f"No locked wording for model '{model_id}', pattern '{pattern}'.")
        index = self._indices[model_id][pattern]
        candidates = WORDING_CANDIDATES_GENERAL[pattern]
        if not 0 <= index < len(candidates):
            raise IndexError(f"Wording index {index} out of range for pattern '{pattern}'.")
        return index

    def build(
        self,
        model_id: str,
        pattern: str,
        method_code: str,
        javadoc: str,
        context: str | None = None,
    ) -> BuiltPrompt:
        """Detection prompt for one method, with the model's locked wording."""
        index = self.wording_index(model_id, pattern)
        if (model_id, pattern) in COLLAPSED:
            warnings.warn(
                f"({model_id}, {pattern}) collapsed to a constant answer in the research "
                f"screening; its verdicts are unreliable.",
                stacklevel=2,
            )
        fn = WORDING_CANDIDATES_GENERAL[pattern][index]
        if pattern in FEWSHOT_PATTERNS:
            if self.fewshot is None:
                raise ValueError(
                    f"Pattern '{pattern}' needs the calibration examples: set "
                    f"'fewshot_file' in the configuration (see scripts/export_fewshot.py)."
                )
            text = fn(method_code, javadoc, self.fewshot.for_pattern(pattern))
        else:
            text = fn(method_code, javadoc)
        return BuiltPrompt(
            text=append_context(text, context, pattern),
            pattern=pattern,
            wording_index=index,
            version=wording_version(pattern, index, fn),
            has_context=bool(context and context.strip()),
        )
