"""MLflow pyfunc input-contract tests."""

import pandas as pd
import pytest

from ecu_assistant.mlflow_model.pyfunc_model import ECUEngineeringAssistantModel


def test_predict_accepts_dataframe_batches():
    model = ECUEngineeringAssistantModel()
    result = model.predict(
        None,
        pd.DataFrame(
            {"query": ["How much RAM does the ECU-850 have?", "Which models support OTA?"]}
        ),
    )

    assert len(result) == 2
    assert "2 GB" in result[0]["answer"]
    assert result[0]["field"] == "memory"
    assert result[0]["citations"][0]["chunk_id"] == "ECU-850-3"


def test_predict_accepts_record_lists():
    model = ECUEngineeringAssistantModel()

    result = model.predict(None, [{"query": "How much RAM does the ECU-850 have?"}])

    assert result[0]["routed_models"] == ["ECU-850"]
    assert result[0]["field"] == "memory"


def test_predict_rejects_missing_query_column():
    with pytest.raises(ValueError, match="query"):
        ECUEngineeringAssistantModel().predict(
            None,
            pd.DataFrame({"question": ["hello"]}),
        )
