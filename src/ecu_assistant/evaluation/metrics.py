"""Evaluation metrics for grounded engineering answers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

STOPWORDS = {
    "a", "all", "also", "an", "and", "are", "as", "at", "be", "by",
    "capabilities", "capability", "capable", "conditions", "dedicated",
    "difference", "different", "for", "from", "has", "have", "in", "is",
    "it", "key", "making", "model", "models", "more", "of", "on", "per",
    "series", "significantly", "specification", "support", "supported",
    "than", "that", "the", "their", "this", "to", "up", "use", "while", "with",
}


def _tokens(text: str) -> set[str]:
    normalized = text.lower().replace("\u63b3c", "°c")
    return {
        token
        for token in re.findall(r"[a-z0-9.+-]+|°c", normalized)
        if token not in STOPWORDS
    }


def answer_recall(answer: str, expected: str) -> float:
    """Measure expected fact-token coverage without requiring identical prose."""

    expected_tokens = _tokens(expected)
    if not expected_tokens:
        return 1.0
    return len(_tokens(answer) & expected_tokens) / len(expected_tokens)


def unique_values(values: Sequence[str]) -> list[str]:
    """Deduplicate labels while preserving order."""

    return list(dict.fromkeys(value for value in values if value))


def set_scores(
    predicted: Sequence[str],
    expected: Sequence[str],
) -> dict[str, float | bool]:
    """Score unordered label sets with exact match, precision, recall, and F1."""

    predicted_set = set(unique_values(predicted))
    expected_set = set(unique_values(expected))
    overlap = predicted_set & expected_set
    precision = 1.0 if not predicted_set else len(overlap) / len(predicted_set)
    recall = 1.0 if not expected_set else len(overlap) / len(expected_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "exact_match": predicted_set == expected_set,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def combined_pass(row: Mapping[str, Any]) -> bool:
    """Return whether all available row-level checks passed."""

    check_names = (
        "answer_correct",
        "routing_correct",
        "retrieval_correct",
        "citation_correct",
        "abstention_correct",
    )
    checks = [row[name] for name in check_names if row.get(name) is not None]
    return all(bool(check) for check in checks)


def _known_values(rows: list[dict[str, Any]], key: str) -> list[Any]:
    return [row[key] for row in rows if row.get(key) is not None]


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    values = _known_values(rows, key)
    return mean(bool(value) for value in values) if values else 0.0


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    values = _known_values(rows, key)
    return mean(float(value) for value in values) if values else 0.0


def _binary_metrics(
    rows: list[dict[str, Any]],
    expected_key: str,
    predicted_key: str,
) -> dict[str, float | int]:
    labelled = [
        (bool(row[expected_key]), bool(row[predicted_key]))
        for row in rows
        if row.get(expected_key) is not None and row.get(predicted_key) is not None
    ]
    true_positive = sum(expected and predicted for expected, predicted in labelled)
    false_positive = sum(not expected and predicted for expected, predicted in labelled)
    false_negative = sum(expected and not predicted for expected, predicted in labelled)
    true_negative = sum(not expected and not predicted for expected, predicted in labelled)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    accuracy = (
        (true_positive + true_negative) / len(labelled)
        if labelled
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
        "support": len(labelled),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate answer, routing, retrieval, citation, abstention, and latency metrics."""

    if not rows:
        return {
            "questions": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "answer_accuracy": 0.0,
            "mean_fact_recall": 0.0,
            "routing_accuracy": 0.0,
            "model_routing_accuracy": 0.0,
            "intent_accuracy": 0.0,
            "field_accuracy": 0.0,
            "retrieval_hit_rate": 0.0,
            "retrieval_mean_precision": 0.0,
            "retrieval_mean_recall": 0.0,
            "citation_exact_match_rate": 0.0,
            "citation_mean_precision": 0.0,
            "citation_mean_recall": 0.0,
            "citation_mean_f1": 0.0,
            "abstention_accuracy": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "abstention_f1": 0.0,
            "mean_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
        }
    latencies = sorted(float(row["latency_seconds"]) for row in rows)
    abstention = _binary_metrics(rows, "expected_abstain", "predicted_abstain")
    return {
        "questions": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "pass_rate": mean(bool(row["passed"]) for row in rows),
        "answer_accuracy": _rate(rows, "answer_correct"),
        "mean_fact_recall": mean(float(row["fact_recall"]) for row in rows),
        "routing_accuracy": _rate(rows, "routing_correct"),
        "model_routing_accuracy": _rate(rows, "model_routing_correct"),
        "intent_accuracy": _rate(rows, "intent_correct"),
        "field_accuracy": _rate(rows, "field_correct"),
        "retrieval_hit_rate": _rate(rows, "retrieval_correct"),
        "retrieval_mean_precision": _mean_key(rows, "retrieval_precision"),
        "retrieval_mean_recall": _mean_key(rows, "retrieval_recall"),
        "citation_exact_match_rate": _rate(rows, "citation_correct"),
        "citation_mean_precision": _mean_key(rows, "citation_precision"),
        "citation_mean_recall": _mean_key(rows, "citation_recall"),
        "citation_mean_f1": _mean_key(rows, "citation_f1"),
        "abstention_accuracy": float(abstention["accuracy"]),
        "abstention_precision": float(abstention["precision"]),
        "abstention_recall": float(abstention["recall"]),
        "abstention_f1": float(abstention["f1"]),
        "abstention_true_positive": int(abstention["true_positive"]),
        "abstention_false_positive": int(abstention["false_positive"]),
        "abstention_false_negative": int(abstention["false_negative"]),
        "abstention_true_negative": int(abstention["true_negative"]),
        "abstention_support": int(abstention["support"]),
        "mean_latency_seconds": mean(latencies),
        "p95_latency_seconds": latencies[max(0, int(len(latencies) * 0.95) - 1)],
    }
