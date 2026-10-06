"""
Regeneration wordings, copied verbatim from the research code
(``src/mitigation/wordings_regeneration.py`` of *Rewriting the Narrative*).

The model is asked for a complete corrected Javadoc; the tool then keeps
only the fields that must change (see ``docdebt.core.javadoc_edit``). The
text of each wording is the one screened in the research: never edit an
existing wording, add new ones at the end of their list.

Mitigation has no persona pattern in this file: only zeroshot, oneshot,
threeshot and cot exist. The core instruction is the "Structured Audit"
tier (audit field by field, output the original unchanged if nothing is wrong).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

WordingFn = Callable[..., str]


@dataclass
class RegenExample:
    method_code: str
    stale_doc: str
    repaired_doc: str  # the correct, ideal full Javadoc block for this example


def _core_instructions() -> str:
    return """You are repairing a Java method's documentation comment. Some parts of
the comment may already be correct; others may be missing, wrong, or
outdated relative to the method's real signature and behavior.

Your job is NOT to write a new comment from scratch. It is to produce a
corrected comment that keeps every part that is already accurate, and
changes only the parts that are missing, wrong, or outdated.

First, silently audit the comment field by field against the code:
- Does every real parameter have a matching @param tag, correctly named?
- If the method returns a value, is there an accurate @return tag?
- Does the description match what the method actually does?
Only after this audit, produce the corrected comment. If your audit finds
no problems at all, output the original comment exactly as given, unchanged."""


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------


def _zeroshot_direct(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block, nothing else."
    )


def _zeroshot_task_framed(method_code: str, stale_doc: str) -> str:
    return (
        "Your task is to repair a Java method's documentation. The comment may "
        "already be partly or fully correct -- your job is to preserve every "
        "accurate part and fix only what is wrong, not to rewrite from "
        "scratch. Audit the comment field by field against the real "
        "signature and behavior before deciding what to change.\n\n"
        f"METHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only the corrected Javadoc comment block."
    )


def _zeroshot_conditions_first(method_code: str, stale_doc: str) -> str:
    return (
        "A repaired Java documentation comment is correct only if every real "
        "parameter has a matching, accurate @param tag, the @return tag (if "
        "required) is accurate, and the description matches the method's "
        "real behavior. Any part of the comment that already satisfies "
        "these conditions must be kept exactly as written -- only parts "
        "that fail must be changed.\n\n"
        f"Repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block."
    )


ZEROSHOT_WORDINGS: list[WordingFn] = [_zeroshot_direct, _zeroshot_task_framed, _zeroshot_conditions_first]


# ---------------------------------------------------------------------------
# One-shot / three-shot
# ---------------------------------------------------------------------------


def _format_example(ex: RegenExample) -> str:
    return (
        f"METHOD CODE:\n{ex.method_code}\n\nCURRENT DOCUMENTATION:\n{ex.stale_doc}\n\n"
        f"Corrected documentation:\n{ex.repaired_doc}"
    )


def _oneshot_direct(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        f"{_core_instructions()}\n\n{example_block}\n\n---\n\n"
        f"Now repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block, nothing else."
    )


def _oneshot_task_framed(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        "Your task is to repair Java documentation, preserving already-correct "
        "content and fixing only what is wrong. Here is a worked example:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only the corrected Javadoc comment block."
    )


def _oneshot_conditions_first(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        "A repaired comment is correct only if every tag matches the real "
        "signature and the description matches real behavior; anything "
        "already correct must be kept unchanged. Here is one worked "
        "example:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block."
    )


ONESHOT_WORDINGS = [_oneshot_direct, _oneshot_task_framed, _oneshot_conditions_first]


def _threeshot_direct(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        f"{_core_instructions()}\n\n{example_block}\n\n---\n\n"
        f"Now repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block, nothing else."
    )


def _threeshot_task_framed(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        "Your task is to repair Java documentation, preserving already-correct "
        "content and fixing only what is wrong. Here are worked examples "
        "covering different kinds of fixes:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only the corrected Javadoc comment block."
    )


def _threeshot_conditions_first(method_code: str, stale_doc: str, examples: list[RegenExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        "A repaired comment is correct only if every tag matches the real "
        "signature and the description matches real behavior; anything "
        "already correct must be kept unchanged. Here are worked "
        "examples:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Repair this case:\n\nMETHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Output only the corrected Javadoc comment block."
    )


THREESHOT_WORDINGS = [_threeshot_direct, _threeshot_task_framed, _threeshot_conditions_first]


# ---------------------------------------------------------------------------
# Chain-of-thought
# ---------------------------------------------------------------------------


def _cot_direct(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "First, briefly audit each field (2-4 sentences): does every "
        "parameter have a correct @param tag, is @return accurate if "
        "needed, does the description match the code? Then, on its own "
        "final block, output exactly:\n"
        "Final Answer:\n<the corrected Javadoc comment block>"
    )


def _cot_structured_checklist(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Work through this checklist explicitly:\n"
        "1. List the method's real parameters and whether it returns a value.\n"
        "2. For each existing @param/@return tag, judge in one short phrase "
        "whether it is accurate and complete.\n"
        "3. For any missing tag, note that it must be written.\n"
        "4. Judge in one short phrase whether the description matches the "
        "method's real behavior.\n\n"
        "After the checklist, on its own final block, output exactly:\n"
        "Final Answer:\n<the corrected Javadoc comment block>"
    )


def _cot_freeform(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nCURRENT DOCUMENTATION:\n{stale_doc}\n\n"
        "Let's think step by step: what are the real parameters, does the "
        "method return a value, do the existing tags match, and does the "
        "description match what the code actually does? Walk through this "
        "reasoning, then, on its own final block, output exactly:\n"
        "Final Answer:\n<the corrected Javadoc comment block>"
    )


COT_WORDINGS: list[WordingFn] = [_cot_direct, _cot_structured_checklist, _cot_freeform]


WORDING_CANDIDATES_REGENERATION: dict[str, list] = {
    "zeroshot": ZEROSHOT_WORDINGS,
    "oneshot": ONESHOT_WORDINGS,
    "threeshot": THREESHOT_WORDINGS,
    "cot": COT_WORDINGS,
}
