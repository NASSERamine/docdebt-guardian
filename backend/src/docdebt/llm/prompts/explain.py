"""Citation prompt: make the model point at the contradicted claim.

This prompt is NOT part of the research screening; it is new in the tool.
It only runs after the validated detection prompts have voted "debt", and
its output is never trusted as is: every quote is checked against the real
Javadoc and code (``docdebt.detection.citations``) and a semantic finding is
emitted only when both quotes are found. A bad explanation can therefore
remove an alert but never create one.
"""

from __future__ import annotations

import hashlib

EXPLAIN_TEMPLATE = """You are auditing a Java method's documentation. A previous review concluded that \
the documentation may make at least one ACTIVE claim (a name, a type, a value, a behavior, \
a condition) that is CONTRADICTED by the code.

Find every such contradicted claim. Rules:
- CLAIM must be copied character for character from the DOCUMENTATION below (a short \
excerpt, at most one sentence).
- CODE must be copied character for character from the METHOD CODE below (a single line \
or expression).
- A missing tag, a missing detail or documentation that is silent about something is NOT a \
contradicted claim. Do not report it.
- If you find no contradicted claim, answer exactly: NONE

METHOD CODE:
{method_code}

DOCUMENTATION:
{javadoc}

Answer with one block per claim and nothing else, in this exact format:
CLAIM: <excerpt copied from the documentation>
CODE: <excerpt copied from the code>
REASON: <one sentence explaining the contradiction>
FIELD: <description | @param NAME | @return | @throws NAME>
---"""

EXPLAIN_VERSION = "explain/v1/" + hashlib.sha256(EXPLAIN_TEMPLATE.encode("utf-8")).hexdigest()[:8]


def build_explain_prompt(method_code: str, javadoc: str) -> str:
    """Citation prompt for one method."""
    return EXPLAIN_TEMPLATE.format(method_code=method_code, javadoc=javadoc)
