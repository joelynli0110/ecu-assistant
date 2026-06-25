"""Evaluation metric tests."""

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.config import AgentConfig
from ecu_assistant.evaluation.golden_set import GoldenQuestion
from ecu_assistant.evaluation.metrics import aggregate_metrics
from ecu_assistant.evaluation.run_eval import evaluate_golden_set, evaluate_response


def test_pass_requires_routing_and_citations_not_only_fact_recall():
    item = GoldenQuestion(
        question_id="unit-1",
        category="Unit",
        question="How much RAM does the ECU-850 have?",
        expected_answer="The ECU-850 has 2 GB LPDDR4 RAM.",
        evaluation_criteria="Unit test",
        expected_models=("ECU-850",),
        expected_intent="specification",
        expected_field="memory",
        expected_citation_chunk_ids=("ECU-850-3",),
        expected_abstain=False,
    )
    response = {
        "answer": "The ECU-850 has 2 GB LPDDR4 RAM.",
        "confidence": 0.99,
        "citations": [{"chunk_id": "ECU-750-2"}],
        "routed_models": ["ECU-750"],
        "intent": "specification",
        "field": "memory",
        "needs_human_review": False,
    }

    row = evaluate_response(item, response, ["ECU-750-2"])

    assert row["answer_correct"] is True
    assert row["routing_correct"] is False
    assert row["retrieval_correct"] is False
    assert row["citation_correct"] is False
    assert row["passed"] is False


def test_high_confidence_error_rate_counts_failed_high_confidence_rows():
    rows = [
        {
            "passed": False,
            "confidence": 0.99,
            "fact_recall": 1.0,
            "latency_seconds": 0.01,
            "answer_correct": True,
            "routing_correct": False,
            "retrieval_correct": False,
            "citation_correct": False,
            "expected_abstain": False,
            "predicted_abstain": False,
        },
        {
            "passed": True,
            "confidence": 0.98,
            "fact_recall": 1.0,
            "latency_seconds": 0.01,
            "answer_correct": True,
            "routing_correct": True,
            "retrieval_correct": True,
            "citation_correct": True,
            "expected_abstain": False,
            "predicted_abstain": False,
        },
        {
            "passed": False,
            "confidence": 0.35,
            "fact_recall": 0.0,
            "latency_seconds": 0.01,
            "answer_correct": False,
            "routing_correct": True,
            "retrieval_correct": True,
            "citation_correct": False,
            "expected_abstain": False,
            "predicted_abstain": True,
        },
    ]

    metrics = aggregate_metrics(rows)

    assert metrics["high_confidence_count"] == 2
    assert metrics["high_confidence_errors"] == 1
    assert metrics["high_confidence_error_rate"] == 0.5
    assert metrics["high_confidence_threshold"] == 0.9


def test_high_confidence_threshold_can_be_overridden_in_metrics():
    rows = [
        {
            "passed": False,
            "confidence": 0.85,
            "fact_recall": 1.0,
            "latency_seconds": 0.01,
            "answer_correct": True,
            "routing_correct": False,
            "retrieval_correct": False,
            "citation_correct": False,
            "expected_abstain": False,
            "predicted_abstain": False,
        }
    ]

    default_metrics = aggregate_metrics(rows)
    custom_metrics = aggregate_metrics(rows, high_confidence_threshold=0.8)

    assert default_metrics["high_confidence_count"] == 0
    assert custom_metrics["high_confidence_threshold"] == 0.8
    assert custom_metrics["high_confidence_count"] == 1
    assert custom_metrics["high_confidence_errors"] == 1


def test_aggregate_metrics_reports_classification_scores():
    answered = GoldenQuestion(
        question_id="unit-2",
        category="Unit",
        question="How much RAM does the ECU-850 have?",
        expected_answer="The ECU-850 has 2 GB LPDDR4 RAM.",
        evaluation_criteria="Unit test",
        expected_models=("ECU-850",),
        expected_intent="specification",
        expected_field="memory",
        expected_citation_chunk_ids=("ECU-850-3",),
        expected_abstain=False,
    )
    abstained = GoldenQuestion(
        question_id="unit-3",
        category="Unit",
        question="Does the ECU-650 support OTA?",
        expected_answer="No retrieved evidence supports a reliable answer to this question.",
        evaluation_criteria="Unit test",
        expected_models=(),
        expected_intent="specification",
        expected_field="ota",
        expected_citation_chunk_ids=(),
        expected_abstain=True,
    )
    rows = [
        evaluate_response(
            answered,
            {
                "answer": "The ECU-850 has 2 GB LPDDR4 RAM.",
                "confidence": 0.99,
                "citations": [{"chunk_id": "ECU-850-3"}],
                "routed_models": ["ECU-850"],
                "intent": "specification",
                "field": "memory",
                "needs_human_review": False,
            },
            ["ECU-850-3"],
        ),
        evaluate_response(
            abstained,
            {
                "answer": "No retrieved evidence supports a reliable answer to this question.",
                "confidence": 0.35,
                "citations": [],
                "routed_models": [],
                "intent": "specification",
                "field": "ota",
                "needs_human_review": True,
            },
            [],
        ),
    ]
    for row in rows:
        row["latency_seconds"] = 0.01

    metrics = aggregate_metrics(rows)

    assert metrics["routing_accuracy"] == 1
    assert metrics["retrieval_hit_rate"] == 1
    assert metrics["citation_exact_match_rate"] == 1
    assert metrics["abstention_accuracy"] == 1
    assert metrics["abstention_precision"] == 1
    assert metrics["abstention_recall"] == 1
    assert metrics["abstention_f1"] == 1
    assert metrics["high_confidence_error_rate"] == 0


def test_golden_evaluation_outputs_classification_metrics():
    result = evaluate_golden_set(ECUEngineeringAgent())
    metrics = result["metrics"]

    assert metrics["questions"] == 13
    assert metrics["passed"] == 13
    assert metrics["routing_accuracy"] == 1
    assert metrics["retrieval_hit_rate"] == 1
    assert metrics["citation_exact_match_rate"] == 1
    assert metrics["abstention_true_positive"] == 3
    assert metrics["abstention_support"] == 13
    assert metrics["high_confidence_errors"] == 0
    assert metrics["high_confidence_error_rate"] == 0
    assert metrics["high_confidence_threshold"] == 0.9


def test_golden_evaluation_uses_agent_configured_high_confidence_threshold():
    agent = ECUEngineeringAgent(AgentConfig(high_confidence_threshold=0.8))

    result = evaluate_golden_set(agent)

    assert result["metrics"]["high_confidence_threshold"] == 0.8
    assert all(
        row["high_confidence"] == (row["confidence"] >= 0.8)
        for row in result["rows"]
    )
