"""Few-shot calibration examples.

The research held out six calibration items (two structural debt, two
semantic debt, two clean) and built the few-shot prompts from them in a
fixed order: one-shot uses items ``[0, 4]`` (one debt example and one clean
example), three-shot uses ``[0, 2, 4]``. The same order must be kept here,
otherwise the one-shot and three-shot prompts are not the ones that were
screened.

The items are not bundled with the code: export them once from the
research dataset with ``scripts/export_fewshot.py`` (you give the six
``item_id`` values in their research order) and point the configuration
at the resulting JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from docdebt.llm.prompts.detection_wordings import GeneralExample

ONESHOT_INDICES = (0, 4)
THREESHOT_INDICES = (0, 2, 4)
EXPECTED_COUNT = 6
_VALID_VERDICTS = {"COMPLETE", "INCOMPLETE"}


@dataclass
class FewShotSet:
    """The six ordered calibration examples."""

    examples: list[GeneralExample]
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.examples) != EXPECTED_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_COUNT} calibration examples in research order, "
                f"got {len(self.examples)} ({self.source})."
            )
        for i, ex in enumerate(self.examples):
            if ex.correct_verdict not in _VALID_VERDICTS:
                raise ValueError(f"Example {i}: correct_verdict must be COMPLETE or INCOMPLETE.")

    def for_pattern(self, pattern: str) -> list[GeneralExample]:
        if pattern == "oneshot":
            return [self.examples[i] for i in ONESHOT_INDICES]
        if pattern == "threeshot":
            return [self.examples[i] for i in THREESHOT_INDICES]
        return []

    @classmethod
    def load(cls, path: str | Path) -> FewShotSet:
        """Load the JSON list written by ``scripts/export_fewshot.py``."""
        path = Path(path).resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        examples = [
            GeneralExample(
                method_code=row["method_code"],
                stale_doc=row["stale_doc"],
                correct_verdict=row["correct_verdict"],
                debt_type=row.get("debt_type", ""),
            )
            for row in rows
        ]
        return cls(examples=examples, source=str(path))
