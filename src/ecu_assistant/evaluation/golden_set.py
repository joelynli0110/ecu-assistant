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
    expected_models: tuple[str, ...] | None = None
    expected_intent: str | None = None
    expected_field: str | None = None
    expected_citation_chunk_ids: tuple[str, ...] | None = None
    expected_abstain: bool | None = None


def default_golden_set_path() -> Path:
    """Return the packaged golden dataset path."""

    return Path(str(resources.files("ecu_assistant.evaluation").joinpath("golden_questions.csv")))


def _split_labels(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(label.strip() for label in value.split("|") if label.strip())


def _optional_text(row: dict[str, str], key: str) -> str | None:
    if key not in row:
        return None
    value = row[key].strip()
    return value or None


def _optional_labels(row: dict[str, str], key: str) -> tuple[str, ...] | None:
    if key not in row:
        return None
    return _split_labels(row[key])


def _optional_bool(row: dict[str, str], key: str) -> bool | None:
    if key not in row:
        return None
    value = row[key].strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value for {key}: {row[key]}")


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
                expected_models=_optional_labels(row, "Expected_Models"),
                expected_intent=_optional_text(row, "Expected_Intent"),
                expected_field=_optional_text(row, "Expected_Field"),
                expected_citation_chunk_ids=_optional_labels(
                    row,
                    "Expected_Chunk_IDs",
                ),
                expected_abstain=_optional_bool(row, "Expected_Abstain"),
            )
            for row in csv.DictReader(handle)
        ]
