"""Brace-token substitution and JSON parsing for the hybrid Replacement prompt.

Taken from the research ``src/mitigation/batched_replacement.py`` (only the
parts the hybrid mechanism uses): a Javadoc's own ``{@link X}`` / ``{@code X}``
braces are hidden behind collision-safe tokens before the prompt is built, so
the model never confuses them with the braces of the JSON it must answer
with; they are restored after parsing.
"""

from __future__ import annotations

import json
import re

LINK_TOKEN = "\u00a7LINK\u00a7"
CODE_TOKEN = "\u00a7CODE\u00a7"
CLOSE_TOKEN = "\u00a7END\u00a7"


def encode_braces(text):
    """Replace literal Javadoc inline-tag braces with collision-safe
    tokens before the text is shown to the model."""
    if not isinstance(text, str):
        return text
    text = text.replace("{@link", LINK_TOKEN)
    text = text.replace("{@code", CODE_TOKEN)
    text = text.replace("}", CLOSE_TOKEN)
    return text


def decode_braces(text):
    """Reverse encode_braces after the model's response has been parsed."""
    if not isinstance(text, str):
        return text
    text = text.replace(LINK_TOKEN, "{@link")
    text = text.replace(CODE_TOKEN, "{@code")
    text = text.replace(CLOSE_TOKEN, "}")
    return text


def repair_invalid_json_escapes(text):
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def parse_batched_json_patch(raw):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    json_text = match.group(0)
    for attempt in [
        lambda: json.loads(json_text),
        lambda: json.loads(json_text, strict=False),
        lambda: json.loads(repair_invalid_json_escapes(json_text)),
        lambda: json.loads(repair_invalid_json_escapes(json_text), strict=False),
    ]:
        try:
            return attempt()
        except json.JSONDecodeError:
            continue
    return None


def looks_truncated_at_brace(text):
    """Applied AFTER decode_braces. Our own tokens always decode to
    perfectly balanced braces by construction, so any imbalance found
    here means the model wrote a genuinely new, truncated {@link/{@code
    from scratch -- the one residual case encoding cannot prevent."""
    if not isinstance(text, str):
        return False
    text = text.rstrip()
    if text.count("{") != text.count("}"):
        return True
    if re.search(r"\{\s*$", text) or re.search(r"\{@\w*$", text):
        return True
    return False
