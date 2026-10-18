"""Parsing of the ``/docdebt accept|reject|regenerate <short_id> [word]`` slash command.

* ``/docdebt accept 1a2b3c4d``: commit the field-by-field fix (Replacement);
* ``/docdebt accept 1a2b3c4d regenerated``: commit the regeneration instead;
* ``/docdebt regenerate 1a2b3c4d``: compute the regeneration to compare;
* ``/docdebt reject 1a2b3c4d false-positive``: dismiss, with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMAND = re.compile(
    r"/docdebt\s+(accept|reject|regenerate)\s+([0-9a-f]{6,12})(?:[ \t]+([\w-]+))?", re.IGNORECASE
)
_REGENERATED = {"regenerated", "regeneration", "regen", "regenere", "régénéré"}

# Optional reason after a reject: "/docdebt reject 1a2b3c4d false-positive".
_REASONS = {
    "false-positive": "false_positive",
    "false_positive": "false_positive",
    "faux-positif": "false_positive",
    "incorrect": "incorrect_fix",
    "incorrect-fix": "incorrect_fix",
    "incorrecte": "incorrect_fix",
    "manual": "manual",
    "manuel": "manual",
    "other": "other",
    "autre": "other",
}


@dataclass(frozen=True)
class SlashCommand:
    decision: str  # "accept" | "reject" | "regenerate"
    short_id: str
    reason: str | None = None  # a reject's reason (see docdebt.service.store.REJECT_REASONS)
    mechanism: str = "replacement"  # what an accept commits: "replacement" | "regeneration"


def parse_slash_command(comment_body: str) -> SlashCommand | None:
    """The first ``/docdebt accept|reject <short_id>`` found in a comment, or
    ``None`` if the comment does not contain one."""
    match = _COMMAND.search(comment_body)
    if not match:
        return None
    decision = match.group(1).lower()
    word = (match.group(3) or "").lower()
    reason = _REASONS.get(word) if decision == "reject" else None
    mechanism = "regeneration" if decision == "accept" and word in _REGENERATED else "replacement"
    return SlashCommand(decision=decision, short_id=match.group(2).lower(), reason=reason, mechanism=mechanism)
