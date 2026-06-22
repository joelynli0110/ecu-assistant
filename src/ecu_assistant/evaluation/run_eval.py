"""Run golden-set evaluation locally or log it to MLflow."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.config import AgentConfig
from ecu_assistant.evaluation.golden_set import load_golden_set
from ecu_assistant.evaluation.metrics import aggregate_metrics, answer_recall


def evaluate_golden_set(
    agent: ECUEngineeringAgent,
    csv_path: Path | None = None,
    pass_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate all golden questions and return aggregate and row-level metrics."""

    rows: list[dict[str, Any]] = []
    for item in load_golden_set(csv_path):
        started = time.perf_counter()
        response = agent.invoke(item.question)
        latency = time.perf_counter() - started
        recall = answer_recall(response["answer"], item.expected_answer)
        rows.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "answer": response["answer"],
                "expected_answer": item.expected_answer,
                "fact_recall": round(recall, 4),
                "latency_seconds": round(latency, 4),
                "confidence": response["confidence"],
                "needs_human_review": response["needs_human_review"],
                "passed": recall >= pass_threshold,
            }
        )
    return {"metrics": aggregate_metrics(rows), "rows": rows}


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
        mlflow.log_metrics(result["metrics"])
        mlflow.log_artifact(str(output_path))
    print(result["metrics"])


if __name__ == "__main__":
    main()
