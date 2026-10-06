"""Repair prompt registry: locked wording per model and pattern, feedback, versions.

The locked indices come from the research Regeneration screening (0-based
indices into ``wordings.WORDING_CANDIDATES_REGENERATION[pattern]``). The
screening also recorded a persona value per model, but the wordings file has
no persona pattern, so those values are not used here.

Models that were not screened (for example GPT-5.6, Claude) have no locked
wording: give one explicitly (``repair.wording_overrides`` in the
configuration or ``--wording`` on the command line). It is used with a
warning, because it has not been validated.
"""

from __future__ import annotations

import hashlib
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass

from docdebt.repair.fewshot import RepairFewShotSet
from docdebt.repair.wordings import WORDING_CANDIDATES_REGENERATION

PATTERNS = ("zeroshot", "oneshot", "threeshot", "cot")
FEWSHOT_PATTERNS = frozenset({"oneshot", "threeshot"})

LOCKED_REPAIR_WORDINGS: dict[str, dict[str, int]] = {
    "qwen2.5-coder:7b": {"zeroshot": 2, "oneshot": 2, "threeshot": 0, "cot": 0},
    "codegemma:7b": {"zeroshot": 0, "oneshot": 1, "threeshot": 0, "cot": 2},
    "deepseek-coder:6.7b-instruct": {"zeroshot": 1, "oneshot": 1, "threeshot": 0, "cot": 1},
}


@dataclass(frozen=True)
class BuiltRepairPrompt:
    text: str
    pattern: str
    wording_index: int
    version: str
    screened: bool


def append_hints(prompt: str, hints: list[str]) -> str:
    """Tell the model which problems an automated review already found."""
    if not hints:
        return prompt
    lines = "\n".join(f"- {h}" for h in hints)
    return f"{prompt}\n\nAn automated review of the current documentation found these problems:\n{lines}"


def append_feedback(prompt: str, issues: list[str]) -> str:
    """Explain why the previous answer was rejected, for a retry."""
    if not issues:
        return prompt
    lines = "\n".join(f"- {i}" for i in issues)
    return (
        f"{prompt}\n\nYour previous answer was rejected for these reasons:\n{lines}\n"
        "Answer again in the same format and fix these problems."
    )


def wording_version(pattern: str, index: int, fn: Callable[..., str]) -> str:
    digest = hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()[:8]
    return f"repair/{pattern}/w{index}/{digest}"


class RepairRegistry:
    def __init__(
        self,
        fewshot: RepairFewShotSet | None = None,
        overrides: dict[str, dict[str, int]] | None = None,
    ) -> None:
        self.fewshot = fewshot
        self._locked = {m: dict(p) for m, p in LOCKED_REPAIR_WORDINGS.items()}
        self._overrides = {m: dict(p) for m, p in (overrides or {}).items()}

    def resolve(self, model_id: str, pattern: str) -> tuple[int, bool]:
        """Wording index and whether it is a research-screened (locked) one."""
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown repair pattern '{pattern}'. Valid patterns: {PATTERNS}.")
        if pattern in self._overrides.get(model_id, {}):
            index, screened = self._overrides[model_id][pattern], False
        elif pattern in self._locked.get(model_id, {}):
            index, screened = self._locked[model_id][pattern], True
        else:
            raise KeyError(
                f"No screened repair wording for model '{model_id}', pattern '{pattern}'. "
                f"Give one explicitly (0 to {len(WORDING_CANDIDATES_REGENERATION[pattern]) - 1}) with "
                f"--wording or repair.wording_overrides in the configuration."
            )
        if not 0 <= index < len(WORDING_CANDIDATES_REGENERATION[pattern]):
            raise IndexError(f"Wording index {index} out of range for pattern '{pattern}'.")
        return index, screened

    def build(
        self,
        model_id: str,
        pattern: str,
        method_code: str,
        javadoc: str,
        *,
        hints: list[str] | None = None,
        feedback: list[str] | None = None,
    ) -> BuiltRepairPrompt:
        index, screened = self.resolve(model_id, pattern)
        if not screened:
            warnings.warn(
                f"Repair wording {index} for ({model_id}, {pattern}) was not screened in the research.",
                stacklevel=2,
            )
        fn = WORDING_CANDIDATES_REGENERATION[pattern][index]
        if pattern in FEWSHOT_PATTERNS:
            if self.fewshot is None:
                raise ValueError(
                    f"Pattern '{pattern}' needs few-shot examples: set repair.fewshot_file "
                    f"(see scripts/export_repair_fewshot.py)."
                )
            text = fn(method_code, javadoc, self.fewshot.for_pattern(pattern))
        else:
            text = fn(method_code, javadoc)
        text = append_feedback(append_hints(text, hints or []), feedback or [])
        return BuiltRepairPrompt(text, pattern, index, wording_version(pattern, index, fn), screened)
