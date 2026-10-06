"""
Detection prompt wordings, copied verbatim from the research code
(``src/detection/wordings.py`` of *Rewriting the Narrative*).

These are the validated candidates: three wordings per pattern that differ
only in phrasing. Which wording each model uses is decided by the research
screening and locked in ``registry.py``. Do not edit the text of an
existing wording: the benchmark numbers only hold for the exact text that
was screened. To try a new wording, add it at the end of its list, which
gives it a new index and a new prompt version.

Definition used by every wording: documentation is INCOMPLETE if EITHER a
structural gap (missing or misnamed @param, missing @return on a non-void
non-constructor) OR a semantic falseness (an active claim contradicted by
the code) is present. Silence or omission is never semantic debt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

WordingFn = Callable[..., str]


@dataclass
class GeneralExample:
    method_code: str
    stale_doc: str
    correct_verdict: str  # "INCOMPLETE" or "COMPLETE"
    debt_type: str = ""   # "structural", "semantic", or "none" -- for readability only


def _core_instructions() -> str:
    return """You are checking whether a Java method's documentation counts as
COMPLETE and CORRECT, or as DEBT (INCOMPLETE). Either of the following,
on its own, is enough to make it INCOMPLETE -- they do not both need
to be present:

1. STRUCTURAL debt: a real parameter has no matching @param tag by
exact name, or the method is non-void and not a constructor and has
no @return tag.

2. SEMANTIC debt: the documentation makes an ACTIVE claim (a name, a
type, a value, a behavior, a condition) that is CONTRADICTED by the
code. A missing tag or an omitted detail is NOT semantic debt by
itself -- only an active, contradicted claim counts. Silence about
something is never a false claim about it.

If neither defect is present, the documentation is COMPLETE."""


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

def _zeroshot_direct(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word on its own line: COMPLETE or INCOMPLETE."
    )


def _zeroshot_task_framed(method_code: str, stale_doc: str) -> str:
    return (
        "Your task is to audit a Java method's documentation for two kinds of "
        "defect, either one sufficient to flag it: a missing or misnamed "
        "@param/@return tag, or a description that actively contradicts what "
        "the code does. A tag that is merely missing detail, or documentation "
        "that is silent about something, is not by itself a semantic defect.\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only one word: COMPLETE or INCOMPLETE."
    )


def _zeroshot_conditions_first(method_code: str, stale_doc: str) -> str:
    return (
        "A Java method's documentation is COMPLETE only if it has no "
        "structural gap (every real parameter has a matching @param tag, and "
        "a @return tag is present when required) AND makes no active claim "
        "that is contradicted by the code. If either condition fails, it is "
        "INCOMPLETE -- a structural gap alone is enough, and a contradicted "
        "active claim alone is enough. Omission or silence about something "
        "is never itself a contradiction.\n\n"
        f"Evaluate this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Output only one word: COMPLETE or INCOMPLETE."
    )


ZEROSHOT_WORDINGS: list[WordingFn] = [_zeroshot_direct, _zeroshot_task_framed, _zeroshot_conditions_first]


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

def _persona_developer(method_code: str, stale_doc: str) -> str:
    return (
        "You are an experienced Java developer performing a documentation "
        "audit. You flag documentation as INCOMPLETE if EITHER a formal tag "
        "is missing/misnamed OR the description actively contradicts the "
        "code's real behavior -- either alone is enough. A tag that is "
        "merely terse, or a description that is silent about a detail, is "
        "not on its own a defect.\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word: COMPLETE or INCOMPLETE."
    )


def _persona_reviewer(method_code: str, stale_doc: str) -> str:
    return (
        "You are a senior engineer reviewing a pull request's documentation. "
        "You flag it as INCOMPLETE for either a missing/misnamed tag or an "
        "active claim that the code contradicts -- either one is sufficient. "
        "You do not flag documentation merely for being terse or silent "
        "about something it never claimed.\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word: COMPLETE or INCOMPLETE."
    )


def _persona_technical_writer(method_code: str, stale_doc: str) -> str:
    return (
        "You are a technical writer auditing API reference documentation for "
        "two independent defect types: missing/misnamed formal tags, and "
        "descriptions that have drifted into actively contradicting the "
        "method's real behavior. Either defect alone makes the documentation "
        "incomplete; neither is required for the other to count.\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word: COMPLETE or INCOMPLETE."
    )


PERSONA_WORDINGS: list[WordingFn] = [_persona_developer, _persona_reviewer, _persona_technical_writer]


# ---------------------------------------------------------------------------
# One-shot / three-shot -- examples drawn from BOTH debt types separately
# ---------------------------------------------------------------------------

def _format_example(ex: GeneralExample) -> str:
    return (
        f"METHOD CODE:\n{ex.method_code}\n\nDOCUMENTATION:\n{ex.stale_doc}\n\n"
        f"Correct answer: {ex.correct_verdict}"
    )


def _oneshot_direct(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        f"{_core_instructions()}\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now evaluate this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word: COMPLETE or INCOMPLETE."
    )


def _oneshot_task_framed(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        "Your task is to audit Java documentation for two kinds of defect: a "
        "missing/misnamed tag, or an active claim contradicting the code. "
        "Here are worked examples:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now audit this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only one word: COMPLETE or INCOMPLETE."
    )


def _oneshot_conditions_first(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example:\n{_format_example(ex)}" for ex in examples)
    return (
        "Documentation is COMPLETE only if it has no structural gap and "
        "makes no contradicted active claim -- either defect alone makes it "
        "INCOMPLETE. Here is one example of each verdict:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Evaluate this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Output only one word: COMPLETE or INCOMPLETE."
    )


ONESHOT_WORDINGS = [_oneshot_direct, _oneshot_task_framed, _oneshot_conditions_first]


def _threeshot_direct(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        f"{_core_instructions()}\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now evaluate this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with exactly one word: COMPLETE or INCOMPLETE."
    )


def _threeshot_task_framed(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        "Your task is to audit Java documentation for two kinds of defect: a "
        "missing/misnamed tag, or an active claim contradicting the code. "
        "Here are worked examples covering both defect types and both "
        "verdicts:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Now audit this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Respond with only one word: COMPLETE or INCOMPLETE."
    )


def _threeshot_conditions_first(method_code: str, stale_doc: str, examples: list[GeneralExample]) -> str:
    example_block = "\n\n".join(f"Example {i}:\n{_format_example(ex)}" for i, ex in enumerate(examples, 1))
    return (
        "Documentation is COMPLETE only if it has no structural gap and "
        "makes no contradicted active claim -- either defect alone makes it "
        "INCOMPLETE. Here are worked examples:\n\n"
        f"{example_block}\n\n---\n\n"
        f"Evaluate this case:\n\nMETHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Output only one word: COMPLETE or INCOMPLETE."
    )


THREESHOT_WORDINGS = [_threeshot_direct, _threeshot_task_framed, _threeshot_conditions_first]


# ---------------------------------------------------------------------------
# Chain-of-thought
# ---------------------------------------------------------------------------

def _cot_direct(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "First, briefly check (2-4 sentences): are all formal tags present "
        "and correctly named, AND does the documentation make any active "
        "claim contradicted by the code? Then, on its own final line, "
        "respond with exactly:\nFinal Answer: COMPLETE\nor\nFinal Answer: INCOMPLETE"
    )


def _cot_structured_checklist(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Work through this checklist explicitly:\n"
        "1. List the method's real parameters and whether it returns a value.\n"
        "2. Structural check: does every parameter have a matching @param "
        "tag, and is @return present if required?\n"
        "3. Semantic check: read the method's actual logic. Does the "
        "documentation make any active claim (name, type, value, behavior, "
        "condition) that is contradicted by what the code does? An omitted "
        "detail does not count -- only an active, contradicted claim.\n\n"
        "After the checklist, on its own final line, respond with exactly:\n"
        "Final Answer: COMPLETE\nor\nFinal Answer: INCOMPLETE"
    )


def _cot_freeform(method_code: str, stale_doc: str) -> str:
    return (
        f"{_core_instructions()}\n\n"
        f"METHOD CODE:\n{method_code}\n\nDOCUMENTATION:\n{stale_doc}\n\n"
        "Let's think step by step: what are the real parameters, does the "
        "method return a value, do the @param/@return tags match, and -- "
        "reading the method's actual logic -- does the documentation make "
        "any active claim that is contradicted by the code? Walk through "
        "this reasoning, then, on its own final line, respond with exactly:\n"
        "Final Answer: COMPLETE\nor\nFinal Answer: INCOMPLETE"
    )


COT_WORDINGS: list[WordingFn] = [_cot_direct, _cot_structured_checklist, _cot_freeform]


WORDING_CANDIDATES_GENERAL: dict[str, list] = {
    "zeroshot": ZEROSHOT_WORDINGS,
    "persona": PERSONA_WORDINGS,
    "oneshot": ONESHOT_WORDINGS,
    "threeshot": THREESHOT_WORDINGS,
    "cot": COT_WORDINGS,
}
