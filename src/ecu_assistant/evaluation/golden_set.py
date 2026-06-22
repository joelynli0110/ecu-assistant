"""Golden-set data loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class GoldenQuestion:
    """One expected engineering question and answer."""

    question_id: str
    category: str
    question: str
    expected_answer: str
    evaluation_criteria: str


def default_golden_set_path() -> Path:
    """Return the packaged golden dataset path."""

    return Path(str(resources.files("ecu_assistant.evaluation").joinpath("golden_questions.csv")))


def load_golden_set(csv_path: Path | None = None) -> list[GoldenQuestion]:
    """Load golden questions from CSV."""

    path = csv_path or default_golden_set_path()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            GoldenQuestion(
                question_id=row["Question_ID"],
                category=row["Category"],
                question=row["Question"],
                expected_answer=row["Expected_Answer"].replace("\u63b3C", "°C"),
                evaluation_criteria=row["Evaluation_Criteria"],
            )
            for row in csv.DictReader(handle)
        ]
