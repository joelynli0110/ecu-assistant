"""Query-router behavior tests."""

from ecu_assistant.agent.nodes import QueryRouter


def test_routes_single_model_without_prefix():
    route = QueryRouter().route("How much RAM does the 850b have?")

    assert route.models == ["ECU-850b"]
    assert route.intent == "specification"


def test_routes_cross_model_comparison():
    route = QueryRouter().route("Compare the CAN bus of ECU-750 and ECU-850.")

    assert route.models == ["ECU-750", "ECU-850"]
    assert route.intent == "comparison"


def test_routes_fleet_question_to_all_models():
    route = QueryRouter().route("Which ECU models support OTA?")

    assert route.models == ["ECU-750", "ECU-850", "ECU-850b"]
    assert route.intent == "fleet_query"
