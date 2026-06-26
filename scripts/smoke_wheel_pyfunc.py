"""Smoke test an installed wheel and MLflow pyfunc model."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

import ecu_assistant
from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.mlflow_model.log_model import log_model


def _assert_wheel_install() -> None:
    package_path = Path(ecu_assistant.__file__).resolve()
    source_tree = (Path.cwd() / "src").resolve()
    if package_path.is_relative_to(source_tree):
        raise AssertionError(
            f"Expected wheel installation, but imported package from {package_path}."
        )


def _smoke_agent() -> None:
    result = ECUEngineeringAgent().invoke("850 storage")
    if result["routed_models"] != ["ECU-850"]:
        raise AssertionError(f"Unexpected route: {result}")
    if "16 GB eMMC" not in result["answer"]:
        raise AssertionError(f"Unexpected answer: {result}")
    if result["citations"][0]["chunk_id"] != "ECU-850-3":
        raise AssertionError(f"Unexpected citations: {result}")


def _smoke_mlflow_pyfunc() -> None:
    import mlflow

    with tempfile.TemporaryDirectory(
        prefix="ecu-mlflow-smoke-",
        ignore_cleanup_errors=True,
    ) as temp_dir:
        tracking_db = (Path(temp_dir) / "mlflow.db").as_posix()
        os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_db}"
        os.environ["ME_ECU_MLFLOW_EXPERIMENT"] = "ecu-assistant-smoke"
        os.environ.pop("ME_ECU_REGISTERED_MODEL_NAME", None)

        model_uri = log_model()
        loaded = mlflow.pyfunc.load_model(model_uri)
        predictions = loaded.predict(
            pd.DataFrame(
                {
                    "query": [
                        "850 storage",
                        "What is the ECU-750 Bluetooth firmware version?",
                    ]
                }
            )
        )

    if len(predictions) != 2:
        raise AssertionError(f"Expected two predictions, got {len(predictions)}.")
    if "16 GB eMMC" not in predictions[0]["answer"]:
        raise AssertionError(f"Unexpected pyfunc answer: {predictions[0]}")
    if predictions[0]["citations"][0]["chunk_id"] != "ECU-850-3":
        raise AssertionError(f"Unexpected pyfunc citation: {predictions[0]}")
    if predictions[1]["needs_human_review"] is not True:
        raise AssertionError(f"Expected review for unknown fact: {predictions[1]}")
    if predictions[1]["citations"] != []:
        raise AssertionError(f"Expected no citations for unknown fact: {predictions[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-wheel-install",
        action="store_true",
        help="Fail if ecu_assistant is imported from the repository src tree.",
    )
    args = parser.parse_args()

    if args.require_wheel_install:
        _assert_wheel_install()
    _smoke_agent()
    _smoke_mlflow_pyfunc()
    print("Wheel and MLflow pyfunc smoke test passed.")


if __name__ == "__main__":
    main()
