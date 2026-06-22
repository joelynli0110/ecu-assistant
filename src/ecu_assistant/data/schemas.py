"""Domain data structures shared across the assistant."""

from __future__ import annotations

from dataclasses import dataclass

ALL_MODELS = ["ECU-750", "ECU-850", "ECU-850b"]

SOURCE_FILES = {
    "ECU-750": "ECU-700_Series_Manual.md",
    "ECU-850": "ECU-800_Series_Base.md",
    "ECU-850b": "ECU-800_Series_Plus.md",
}


@dataclass(frozen=True)
class ModelRecord:
    """Structured specifications extracted from one source document."""

    model: str
    series: str
    source: str
    specs: dict[str, str]
    ota_supported: bool | None
    text: str


@dataclass(frozen=True)
class RouteDecision:
    """Models and intent selected for a user question."""

    models: list[str]
    intent: str
    reason: str


@dataclass(frozen=True)
class AnswerResult:
    """Answer text plus confidence and review metadata."""

    text: str
    confidence: float
    needs_human_review: bool = False

