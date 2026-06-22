"""Custom MLflow pyfunc wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.config import AgentConfig

try:
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover - core package can run without MLflow
    class PythonModel:  # type: ignore[no-redef]
        """Compatibility base when MLflow is not installed."""


def extract_queries(model_input: Any) -> list[str]:
    """Normalize supported pyfunc input shapes into query strings."""

    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, pd.DataFrame):
        if "query" not in model_input.columns:
            raise ValueError("Model input DataFrame must contain a 'query' column.")
        return model_input["query"].astype(str).tolist()
    if isinstance(model_input, dict):
        value = model_input.get("query")
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value]
    if isinstance(model_input, Iterable):
        values = list(model_input)
        if all(isinstance(item, str) for item in values):
            return values
        if all(isinstance(item, dict) and "query" in item for item in values):
            return [str(item["query"]) for item in values]
    raise ValueError(
        "Model input must be a query string, a DataFrame with 'query', "
        "a {'query': ...} mapping, or a list of query records."
    )


class ECUEngineeringAssistantModel(PythonModel):
    """MLflow-compatible model that lazily initializes the LangGraph agent."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config
        self.agent: ECUEngineeringAgent | None = None

    def load_context(self, context: Any) -> None:
        """Initialize the agent using MLflow's packaged document artifact."""

        docs_dir = None
        if context is not None and getattr(context, "artifacts", None):
            docs_dir = context.artifacts.get("docs")
        if self.config:
            config = replace(
                self.config,
                docs_dir=Path(docs_dir) if docs_dir else self.config.docs_dir,
            )
        else:
            config = AgentConfig.from_env(docs_dir)
        self.agent = ECUEngineeringAgent(config)

    def predict(self, context, model_input, params=None):
        """Answer one or more queries using MLflow's pyfunc contract."""

        del params
        if self.agent is None:
            self.load_context(context)
        assert self.agent is not None
        return [self.agent.invoke(query) for query in extract_queries(model_input)]

