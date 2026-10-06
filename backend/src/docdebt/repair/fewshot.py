"""Few-shot examples for the repair prompts (one-shot and three-shot patterns).

The JSON file lists the examples and states explicitly which ones each
pattern uses, so nothing depends on a guessed order::

    {"examples": [{"method_code": "...", "stale_doc": "...", "repaired_doc": "..."}, ...],
     "oneshot": [0], "threeshot": [0, 1, 2]}

Create it with ``scripts/export_repair_fewshot.py``. Zero-shot and
chain-of-thought need no examples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docdebt.repair.wordings import RegenExample


@dataclass
class RepairFewShotSet:
    examples: list[RegenExample]
    oneshot: list[int]
    threeshot: list[int]
    source: str = ""

    def __post_init__(self) -> None:
        for name, indices in (("oneshot", self.oneshot), ("threeshot", self.threeshot)):
            bad = [i for i in indices if not 0 <= i < len(self.examples)]
            if bad:
                raise ValueError(f"{name} indices {bad} are out of range ({len(self.examples)} examples).")

    def for_pattern(self, pattern: str) -> list[RegenExample]:
        if pattern == "oneshot":
            return [self.examples[i] for i in self.oneshot]
        if pattern == "threeshot":
            return [self.examples[i] for i in self.threeshot]
        return []

    @classmethod
    def load(cls, path: str | Path) -> RepairFewShotSet:
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        examples = [RegenExample(r["method_code"], r["stale_doc"], r["repaired_doc"]) for r in data["examples"]]
        return cls(examples, list(data["oneshot"]), list(data["threeshot"]), str(path))
