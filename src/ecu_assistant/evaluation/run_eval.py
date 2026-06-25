"""Run golden-set evaluation locally or log it to MLflow."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.config import AgentConfig
from ecu_assistant.evaluation.golden_set import GoldenQuestion, load_golden_set
from ecu_assistant.evaluation.metrics import (
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    aggregate_metrics,
    answer_recall,
    combined_pass,
    set_scores,
    unique_values,
)
from ecu_assistant.reproducibility import (
    build_reproducibility_metadata,
    flatten_reproducibility_params,
)


def _run_agent_with_trace(
    agent: ECUEngineeringAgent,
    query: str,
) -> tuple[dict[str, Any], list[str]]:
    """Run the agent and return both the public response and retrieved chunk IDs."""

    clean_query = query.strip()
    if not clean_query:
        return agent.invoke(query), []
    state = agent.graph.invoke({"query": clean_query})
    response = {
        "answer": state["answer"],
        "confidence": round(float(state["confidence"]), 3),
        "citations": state["citations"],
        "routed_models": state["routed_models"],
        "intent": state["intent"],
        "field": state.get("field"),
        "needs_human_review": state["needs_human_review"],
        "review_reason": state.get("review_reason", "grounded"),
    }
    retrieved_chunk_ids = unique_values(
        [
            document.metadata["chunk_id"]
            for document in state.get("documents", [])
        ]
    )
    return response, retrieved_chunk_ids


def _routing_scores(
    item: GoldenQuestion,
    response: dict[str, Any],
) -> dict[str, bool | None]:
    routing_labelled = item.expected_models is not None or item.expected_intent is not None
    model_correct = (
        response["routed_models"] == list(item.expected_models)
        if item.expected_models is not None
        else None
    )
    intent_correct = (
        response["intent"] == item.expected_intent
        if item.expected_intent is not None
        else None
    )
    field_correct = (
        response.get("field") == item.expected_field
        if routing_labelled
        else None
    )
    checks = [
        check
        for check in (model_correct, intent_correct, field_correct)
        if check is not None
    ]
    return {
        "model_routing_correct": model_correct,
        "intent_correct": intent_correct,
        "field_correct": field_correct,
        "routing_correct": all(checks) if checks else None,
    }


def _retrieval_scores(
    expected_chunk_ids: tuple[str, ...] | None,
    retrieved_chunk_ids: list[str],
) -> dict[str, float | bool | None]:
    if expected_chunk_ids is None or not expected_chunk_ids:
        return {
            "retrieval_correct": None,
            "retrieval_precision": None,
            "retrieval_recall": None,
        }
    scores = set_scores(retrieved_chunk_ids, expected_chunk_ids)
    return {
        "retrieval_correct": scores["recall"] == 1.0,
        "retrieval_precision": scores["precision"],
        "retrieval_recall": scores["recall"],
    }


def _citation_scores(
    expected_chunk_ids: tuple[str, ...] | None,
    citation_chunk_ids: list[str],
) -> dict[str, float | bool | None]:
    if expected_chunk_ids is None:
        return {
            "citation_correct": None,
            "citation_precision": None,
            "citation_recall": None,
            "citation_f1": None,
        }
    scores = set_scores(citation_chunk_ids, expected_chunk_ids)
    return {
        "citation_correct": bool(scores["exact_match"]),
        "citation_precision": scores["precision"],
        "citation_recall": scores["recall"],
        "citation_f1": scores["f1"],
    }


def evaluate_response(
    item: GoldenQuestion,
    response: dict[str, Any],
    retrieved_chunk_ids: list[str],
    answer_recall_threshold: float = 0.50,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one response against answer, routing, evidence, and abstention labels."""

    recall = answer_recall(response["answer"], item.expected_answer)
    citation_chunk_ids = unique_values(
        [citation["chunk_id"] for citation in response["citations"]]
    )
    predicted_abstain = bool(response["needs_human_review"])
    abstention_correct = (
        predicted_abstain == item.expected_abstain
        if item.expected_abstain is not None
        else None
    )
    row: dict[str, Any] = {
        "question_id": item.question_id,
        "category": item.category,
        "question": item.question,
        "answer": response["answer"],
        "expected_answer": item.expected_answer,
        "fact_recall": round(recall, 4),
        "answer_correct": recall >= answer_recall_threshold,
        "confidence": response["confidence"],
        "high_confidence": response["confidence"] >= high_confidence_threshold,
        "needs_human_review": response["needs_human_review"],
        "review_reason": response.get("review_reason", "grounded"),
        "expected_models": (
            list(item.expected_models) if item.expected_models is not None else None
        ),
        "routed_models": response["routed_models"],
        "expected_intent": item.expected_intent,
        "intent": response["intent"],
        "expected_field": item.expected_field,
        "field": response.get("field"),
        "expected_chunk_ids": (
            list(item.expected_citation_chunk_ids)
            if item.expected_citation_chunk_ids is not None
            else None
        ),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "citation_chunk_ids": citation_chunk_ids,
        "expected_abstain": item.expected_abstain,
        "predicted_abstain": predicted_abstain,
        "abstention_correct": abstention_correct,
    }
    row.update(_routing_scores(item, response))
    row.update(_retrieval_scores(item.expected_citation_chunk_ids, retrieved_chunk_ids))
    row.update(_citation_scores(item.expected_citation_chunk_ids, citation_chunk_ids))
    row["passed"] = combined_pass(row)
    return row


def evaluate_golden_set(
    agent: ECUEngineeringAgent,
    csv_path: Path | None = None,
    answer_recall_threshold: float = 0.50,
    pass_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate all golden questions and return aggregate and row-level metrics."""

    if pass_threshold is not None:
        answer_recall_threshold = pass_threshold
    rows: list[dict[str, Any]] = []
    for item in load_golden_set(csv_path):
        started = time.perf_counter()
        response, retrieved_chunk_ids = _run_agent_with_trace(agent, item.question)
        latency = time.perf_counter() - started
        row = evaluate_response(
            item,
            response,
            retrieved_chunk_ids,
            answer_recall_threshold,
            agent.config.high_confidence_threshold,
        )
        row["latency_seconds"] = round(latency, 4)
        rows.append(row)
    metadata = build_reproducibility_metadata(agent.config, csv_path)
    return {
        "metrics": aggregate_metrics(
            rows,
            high_confidence_threshold=agent.config.high_confidence_threshold,
        ),
        "metadata": metadata,
        "rows": rows,
    }


def write_evaluation(result: dict[str, Any], output_path: Path) -> None:
    """Persist evaluation output as readable JSON."""

    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run the golden set locally."""

    data_path = os.getenv("ME_ECU_EVAL_DATA")
    output_path = Path(os.getenv("ME_ECU_EVAL_OUTPUT", "evaluation-results.json"))
    result = evaluate_golden_set(
        ECUEngineeringAgent(AgentConfig.from_env()),
        Path(data_path) if data_path else None,
    )
    write_evaluation(result, output_path)
    print(result["metrics"])
    print(f"Detailed results: {output_path.resolve()}")


def mlflow_main() -> None:
    """Run the golden set and log metrics and details to MLflow."""

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MLflow evaluation requires `pip install -e .[mlflow]`."
        ) from exc
    data_path = os.getenv("ME_ECU_EVAL_DATA")
    output_path = Path(os.getenv("ME_ECU_EVAL_OUTPUT", "evaluation-results.json"))
    result = evaluate_golden_set(
        ECUEngineeringAgent(AgentConfig.from_env()),
        Path(data_path) if data_path else None,
    )
    write_evaluation(result, output_path)
    mlflow.set_experiment(os.getenv("ME_ECU_MLFLOW_EXPERIMENT", "ecu-assistant"))
    with mlflow.start_run(run_name="evaluate-ecu-assistant"):
        reproducibility_params = flatten_reproducibility_params(result["metadata"])
        mlflow.log_metrics(result["metrics"])
        mlflow.log_params(reproducibility_params)
        mlflow.set_tags(
            {
                "task": "evaluation",
                "domain": "automotive-ecu",
                "package": "ecu-assistant",
                "git_sha": reproducibility_params["git_sha"],
                "document_hash": reproducibility_params["document_hash"],
                "evaluation_set_version": reproducibility_params[
                    "evaluation_set_version"
                ],
            }
        )
        mlflow.log_dict(result["metadata"], "reproducibility.json")
        mlflow.log_dict(result["metadata"]["config"], "config.json")
        mlflow.log_dict(result["metadata"]["documents"], "document_manifest.json")
        mlflow.log_dict(result["metadata"]["evaluation_set"], "evaluation_set.json")
        mlflow.log_artifact(str(output_path))
    print(result["metrics"])


if __name__ == "__main__":
    main()
