"""Log and optionally register the custom MLflow model."""

from __future__ import annotations

import os
from inspect import signature as inspect_signature
from pathlib import Path

from ecu_assistant import __version__
from ecu_assistant.config import AgentConfig
from ecu_assistant.mlflow_model.pyfunc_model import ECUEngineeringAssistantModel


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MLflow is required for model logging. Install `ecu-assistant[mlflow]`."
        ) from exc
    return mlflow


def _provider_requirements(config: AgentConfig) -> list[str]:
    requirements: list[str] = []
    providers = {config.llm_provider, config.embedding_provider}
    if "openai" in providers:
        requirements.append("langchain-openai>=0.3,<2")
    if "anthropic" in providers:
        requirements.append("langchain-anthropic>=0.3,<2")
    if "ollama" in providers:
        requirements.append("langchain-ollama>=0.2,<2")
    return requirements


def log_model() -> str:
    """Log the LangGraph assistant as an MLflow pyfunc model."""

    mlflow = _require_mlflow()
    import pandas as pd
    from mlflow.models import infer_signature

    mlflow.set_experiment(os.getenv("ME_ECU_MLFLOW_EXPERIMENT", "ecu-assistant"))
    registered_name = os.getenv("ME_ECU_REGISTERED_MODEL_NAME")
    config = AgentConfig.from_env()
    package_root = Path(__file__).resolve().parents[1]
    input_example = pd.DataFrame({"query": ["How much RAM does the ECU-850 have?"]})
    output_example = [
        {
            "answer": "The ECU-850 has 2 GB LPDDR4 RAM.",
            "confidence": 0.99,
            "citations": ["ECU-800_Series_Base.md"],
            "routed_models": ["ECU-850"],
            "intent": "specification",
            "needs_human_review": False,
        }
    ]

    with mlflow.start_run(run_name="build-ecu-assistant") as run:
        mlflow.log_params(
            {
                "agent_framework": "langgraph",
                "retrieval_k": config.retrieval_k,
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model or "extractive-fallback",
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model or "local-hash",
                "package_version": __version__,
            }
        )
        mlflow.set_tags(
            {
                "task": "question-answering",
                "domain": "automotive-ecu",
                "package": "ecu-assistant",
            }
        )
        arguments = {
            "python_model": ECUEngineeringAssistantModel(config),
            "artifacts": {"docs": str(package_root / "data" / "documents")},
            "code_paths": [str(package_root)],
            "input_example": input_example,
            "signature": infer_signature(input_example, output_example),
            "pip_requirements": [
                "langchain-core>=0.3,<2",
                "langgraph>=0.2,<2",
                "numpy>=1.26,<3",
                "pandas>=2,<3",
                *_provider_requirements(config),
            ],
            "registered_model_name": registered_name or None,
        }
        key = (
            "name"
            if "name" in inspect_signature(mlflow.pyfunc.log_model).parameters
            else "artifact_path"
        )
        arguments[key] = "model"
        model_info = mlflow.pyfunc.log_model(**arguments)
        mlflow.set_tag("model_uri", model_info.model_uri)
        print(f"MLflow run: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        return model_info.model_uri


def main() -> None:
    """CLI entry point."""

    log_model()


if __name__ == "__main__":
    main()
