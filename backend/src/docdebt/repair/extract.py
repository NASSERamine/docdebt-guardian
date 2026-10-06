"""Extract the Javadoc block from a raw model answer.

Models wrap the comment in ```java fences, add prose before or after it, or
(chain-of-thought) reason first and give the comment after ``Final Answer:``.
"""

from __future__ import annotations

import re

_BLOCK = re.compile(r"/\*\*.*?\*/", re.DOTALL)
_FINAL = "final answer:"


def extract_javadoc_block(raw: str | None) -> str | None:
    """The last ``/** ... */`` block of the answer (of the text after the last
    ``Final Answer:`` when present), or ``None`` when there is none."""
    if not raw:
        return None
    index = raw.lower().rfind(_FINAL)
    candidates = [raw[index + len(_FINAL) :]] if index != -1 else []
    candidates.append(raw)
    for text in candidates:
        matches = _BLOCK.findall(text)
        if matches:
            return matches[-1]
    return None
