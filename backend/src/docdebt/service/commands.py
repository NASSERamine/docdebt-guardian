"""Parsing of the ``/docdebt accept|reject <short_id>`` slash command."""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMAND = re.compile(r"/docdebt\s+(accept|reject)\s+([0-9a-f]{6,12})", re.IGNORECASE)


@dataclass(frozen=True)
class SlashCommand:
    decision: str  # "accept" | "reject"
    short_id: str


def parse_slash_command(comment_body: str) -> SlashCommand | None:
    """The first ``/docdebt accept|reject <short_id>`` found in a comment, or
    ``None`` if the comment does not contain one."""
    match = _COMMAND.search(comment_body)
    if not match:
        return None
    return SlashCommand(decision=match.group(1).lower(), short_id=match.group(2).lower())
