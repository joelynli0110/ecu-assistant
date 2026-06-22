"""Evaluation metrics for grounded engineering answers."""

from __future__ import annotations

import re
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
    normalized = text.lower().replace("掳c", "°c")
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


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate pass rate, recall, and latency metrics."""

    if not rows:
        return {
            "questions": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "mean_fact_recall": 0.0,
            "mean_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
        }
    latencies = sorted(float(row["latency_seconds"]) for row in rows)
    return {
        "questions": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "pass_rate": mean(bool(row["passed"]) for row in rows),
        "mean_fact_recall": mean(float(row["fact_recall"]) for row in rows),
        "mean_latency_seconds": mean(latencies),
        "p95_latency_seconds": latencies[max(0, int(len(latencies) * 0.95) - 1)],
    }

