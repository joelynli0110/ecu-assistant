"""Query-router behavior tests."""

from ecu_assistant.agent.nodes import QueryRouter


def test_routes_single_model_without_prefix():
    route = QueryRouter().route("How much RAM does the 850b have?")

    assert route.models == ["ECU-850b"]
    assert route.intent == "specification"
    assert route.field == "memory"


def test_routes_cross_model_comparison():
    route = QueryRouter().route("Compare the CAN bus of ECU-750 and ECU-850.")

    assert route.models == ["ECU-750", "ECU-850"]
    assert route.intent == "comparison"
    assert route.field == "can"


def test_routes_contrast_wording_as_comparison():
    route = QueryRouter().route("Contrast the processors in ECU-850 and ECU-850b.")

    assert route.models == ["ECU-850", "ECU-850b"]
    assert route.intent == "comparison"
    assert route.field == "processor"


def test_routes_fleet_question_to_all_models():
    route = QueryRouter().route("Which ECU models support OTA?")

    assert route.models == ["ECU-750", "ECU-850", "ECU-850b"]
    assert route.intent == "fleet_query"
    assert route.field == "ota"


def test_bare_850_storage_routes_only_to_ecu_850():
    route = QueryRouter().route("850 storage")

    assert route.models == ["ECU-850"]
    assert route.intent == "specification"
    assert route.field == "storage"


def test_bare_750_is_recognized():
    route = QueryRouter().route("750 processor")

    assert route.models == ["ECU-750"]
    assert route.field == "processor"


def test_850b_variants_do_not_match_ecu_850():
    router = QueryRouter()

    for query in ("850b storage", "ECU-850B storage", "ECU 850 b storage"):
        route = router.route(query)
        assert route.models == ["ECU-850b"]
        assert route.field == "storage"


def test_lowercase_can_modal_does_not_route_to_can_bus():
    route = QueryRouter().route("Can ECU-850 update itself wirelessly?")

    assert route.models == ["ECU-850"]
    assert route.intent == "specification"
    assert route.field == "ota"


def test_wrong_explicit_model_does_not_expand_to_all_models():
    route = QueryRouter().route("Does the ECU-650 support OTA?")

    assert route.models == []
    assert route.intent == "specification"
    assert route.field == "ota"


def test_mixed_valid_and_wrong_models_are_not_partially_answered():
    route = QueryRouter().route("Compare storage for ECU-650 and ECU-850.")

    assert route.models == []
    assert route.intent == "comparison"
    assert route.field == "storage"


def test_unknown_variant_does_not_fall_back_to_base_model():
    route = QueryRouter().route("850c storage")

    assert route.models == []
    assert route.intent == "specification"
    assert route.field == "storage"
