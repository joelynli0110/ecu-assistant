"""Confidence and human-review policy tests."""

from ecu_assistant.agent.confidence import ConfidenceInputs, assess_confidence
from ecu_assistant.agent.graph import ECUEngineeringAgent


def test_no_evidence_answer_is_low_confidence_review():
    assessment = assess_confidence(
        ConfidenceInputs(
            answer="No retrieved evidence supports a reliable answer to this question.",
            base_confidence=0.92,
            citations=[],
            routed_models=["ECU-850"],
            upstream_review_required=True,
            low_confidence_threshold=0.55,
        )
    )

    assert assessment.confidence == 0.35
    assert assessment.needs_human_review is True
    assert assessment.review_reason == "no_supporting_evidence"


def test_unsupported_scope_is_low_confidence_review():
    assessment = assess_confidence(
        ConfidenceInputs(
            answer="No retrieved evidence supports a reliable answer to this question.",
            base_confidence=0.99,
            citations=[],
            routed_models=[],
            upstream_review_required=True,
            low_confidence_threshold=0.55,
        )
    )

    assert assessment.confidence == 0.35
    assert assessment.needs_human_review is True
    assert assessment.review_reason == "no_supported_model_scope"


def test_uncited_non_abstention_answer_is_not_high_confidence():
    assessment = assess_confidence(
        ConfidenceInputs(
            answer="The ECU-850 has 2 GB LPDDR4 RAM.",
            base_confidence=0.99,
            citations=[],
            routed_models=["ECU-850"],
            upstream_review_required=False,
            low_confidence_threshold=0.55,
        )
    )

    assert assessment.confidence == 0.54
    assert assessment.needs_human_review is True
    assert assessment.review_reason == "missing_grounding_citations"


def test_grounded_answer_keeps_high_confidence_without_review():
    assessment = assess_confidence(
        ConfidenceInputs(
            answer="The ECU-850 has 2 GB LPDDR4 RAM [ECU-850-3].",
            base_confidence=0.99,
            citations=[{"chunk_id": "ECU-850-3"}],
            routed_models=["ECU-850"],
            upstream_review_required=False,
            low_confidence_threshold=0.55,
        )
    )

    assert assessment.confidence == 0.99
    assert assessment.needs_human_review is False
    assert assessment.review_reason == "grounded"


def test_unknown_fact_is_low_confidence_abstention_end_to_end():
    result = ECUEngineeringAgent().invoke("What is the ECU-750 Bluetooth firmware version?")

    assert result["confidence"] == 0.35
    assert result["needs_human_review"] is True
    assert result["review_reason"] == "no_supporting_evidence"
    assert result["citations"] == []
