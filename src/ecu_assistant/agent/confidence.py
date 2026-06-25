"""Confidence calibration and human-review policy."""

from __future__ import annotations

from dataclasses import dataclass

NO_EVIDENCE_ANSWER = "No retrieved evidence supports a reliable answer to this question."
LOW_EVIDENCE_CONFIDENCE = 0.35


@dataclass(frozen=True)
class ConfidenceInputs:
    """Signals used to calibrate an answer."""

    answer: str
    base_confidence: float
    citations: list[dict[str, str]]
    routed_models: list[str]
    upstream_review_required: bool
    low_confidence_threshold: float


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Calibrated confidence and review decision."""

    confidence: float
    needs_human_review: bool
    review_reason: str


def _cap_below_threshold(confidence: float, threshold: float) -> float:
    return min(confidence, max(0.0, threshold - 0.01))


def is_abstention_answer(answer: str) -> bool:
    """Return whether an answer is the controlled no-evidence response."""

    return answer.strip() == NO_EVIDENCE_ANSWER


def assess_confidence(inputs: ConfidenceInputs) -> ConfidenceAssessment:
    """Calibrate confidence using grounding, routing scope, and review signals."""

    clean_answer = inputs.answer.strip()
    if not clean_answer:
        return ConfidenceAssessment(0.0, True, "empty_answer")

    if not inputs.routed_models:
        return ConfidenceAssessment(
            min(inputs.base_confidence, LOW_EVIDENCE_CONFIDENCE),
            True,
            "no_supported_model_scope",
        )

    if is_abstention_answer(clean_answer):
        return ConfidenceAssessment(
            min(inputs.base_confidence, LOW_EVIDENCE_CONFIDENCE),
            True,
            "no_supporting_evidence",
        )

    if not inputs.citations:
        return ConfidenceAssessment(
            _cap_below_threshold(
                inputs.base_confidence,
                inputs.low_confidence_threshold,
            ),
            True,
            "missing_grounding_citations",
        )

    if inputs.upstream_review_required:
        return ConfidenceAssessment(
            _cap_below_threshold(
                inputs.base_confidence,
                inputs.low_confidence_threshold,
            ),
            True,
            "upstream_review_required",
        )

    return ConfidenceAssessment(inputs.base_confidence, False, "grounded")
