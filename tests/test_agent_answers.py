"""End-to-end tests against the challenge questions."""

import re

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.agent.nodes import GroundedLLMAnswerer
from ecu_assistant.config import AgentConfig
from ecu_assistant.data.loaders import DocumentRepository
from ecu_assistant.retrieval.chunking import chunk_records


@pytest.fixture(scope="module")
def agent():
    return ECUEngineeringAgent()


@pytest.mark.parametrize(
    ("question", "required_terms"),
    [
        ("What is the maximum operating temperature for the ECU-750?", ["+85°C", "-40°C"]),
        ("How much RAM does the ECU-850 have?", ["2 GB", "LPDDR4"]),
        ("What are the AI capabilities of the ECU-850b?", ["NPU", "5 TOPS", "edge AI"]),
        (
            "What are the differences between ECU-850 and ECU-850b?",
            ["5 TOPS", "4 GB", "2 GB", "1.5 GHz", "1.2 GHz"],
        ),
        (
            "Compare the CAN bus capabilities of ECU-750 and ECU-850.",
            ["Single Channel", "1 Mbps", "Dual Channel", "2 Mbps"],
        ),
        ("What is the power consumption of the ECU-850b under load?", ["1.7A", "550mA"]),
        (
            "Which ECU models support Over-the-Air (OTA) updates?",
            ["ECU-850", "ECU-850b", "ECU-750", "does not support"],
        ),
        (
            "How does the storage capacity compare across all ECU models?",
            ["2 MB", "16 GB", "32 GB"],
        ),
        (
            "Which ECU can operate in the harshest temperature conditions?",
            ["ECU-850", "ECU-850b", "+105°C", "+85°C"],
        ),
        (
            "How do you enable the NPU on the ECU-850b?",
            ["me-driver-ctl --enable-npu --mode=performance"],
        ),
    ],
)
def test_challenge_answers(agent, question, required_terms):
    result = agent.invoke(question)

    assert all(term.lower() in result["answer"].lower() for term in required_terms)
    assert result["confidence"] >= 0.9
    assert result["citations"]
    assert result["needs_human_review"] is False


def test_unsupported_question_is_flagged_for_review(agent):
    result = agent.invoke("What is the ECU-750 Bluetooth firmware version?")

    assert result["routed_models"] == ["ECU-750"]
    assert result["needs_human_review"] is True
    assert result["confidence"] < 0.55


def test_grounded_answerer_accepts_a_langchain_chat_model():
    documents = [
        document
        for document in chunk_records(DocumentRepository().records)
        if document.metadata["model"] == "ECU-850"
    ]
    answerer = GroundedLLMAnswerer(
        AgentConfig(),
        model=FakeListChatModel(
            responses=[
                "The ECU-850 has 2 GB LPDDR4 RAM [ECU-850-3]."
            ]
        ),
    )

    result = answerer.answer("How much RAM does the ECU-850 have?", documents, "test")

    assert result.confidence == 0.86
    assert result.needs_human_review is False
    assert result.evidence_chunk_ids == ("ECU-850-3",)


@pytest.mark.parametrize(
    ("question", "required_terms"),
    [
        (
            "Which CPU architecture and clock rate powers the ECU-750?",
            ["ECU-750", "Cortex-M4", "120 MHz"],
        ),
        (
            "How fast is the CAN bus on the 850b?",
            ["ECU-850b", "Dual Channel", "2 Mbps"],
        ),
        (
            "Contrast the flash memory available on ECU-750 and ECU-850b.",
            ["ECU-750", "2 MB", "ECU-850b", "32 GB"],
        ),
        (
            "Can the 750 receive remote firmware updates?",
            ["ECU-750", "does not support OTA"],
        ),
        (
            "What network interface does the ECU-850 use?",
            ["ECU-850", "1x 100BASE-T1"],
        ),
        (
            "Compare the processors used by ECU-850 and ECU-850b.",
            ["ECU-850", "1.2 GHz", "ECU-850b", "1.5 GHz"],
        ),
        (
            "Which physical connectors are available on ECU-750?",
            ["Main Automotive Connector", "JTAG"],
        ),
    ],
)
def test_generic_spec_questions_support_new_wordings(agent, question, required_terms):
    result = agent.invoke(question)

    assert all(term.lower() in result["answer"].lower() for term in required_terms)
    assert result["needs_human_review"] is False


@pytest.mark.parametrize(
    ("question", "expected_model", "expected_field", "expected_value"),
    [
        ("850 storage", "ECU-850", "storage", "16 GB eMMC"),
        ("750 processor", "ECU-750", "processor", "Cortex-M4"),
        ("850b storage", "ECU-850b", "storage", "32 GB eMMC"),
        ("ECU 850 b CAN speed", "ECU-850b", "can", "2 Mbps"),
    ],
)
def test_explicit_model_scope_is_preserved(
    agent,
    question,
    expected_model,
    expected_field,
    expected_value,
):
    result = agent.invoke(question)

    assert result["routed_models"] == [expected_model]
    assert result["intent"] == "specification"
    assert result["field"] == expected_field
    assert expected_value in result["answer"]


def test_each_inline_evidence_marker_has_a_structured_chunk_citation(agent):
    result = agent.invoke("Compare the processors used by ECU-850 and ECU-850b.")

    marker_ids = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", result["answer"]))
    citation_ids = {citation["chunk_id"] for citation in result["citations"]}

    assert marker_ids == {"ECU-850-3", "ECU-850b-2"}
    assert citation_ids == marker_ids
    assert all(
        {"source", "section", "chunk_id", "model"} <= citation.keys()
        for citation in result["citations"]
    )


def test_single_fact_citation_points_to_the_supporting_section(agent):
    result = agent.invoke("850 storage")

    assert result["citations"] == [
        {
            "source": "ECU-800_Series_Base.md",
            "section": "ECU-850 Technical Specifications",
            "chunk_id": "ECU-850-3",
            "model": "ECU-850",
        }
    ]
    assert "[ECU-850-3]" in result["answer"]


def test_derived_can_comparison_cites_both_supporting_chunks(agent):
    result = agent.invoke("Compare the CAN bus capabilities of ECU-750 and ECU-850.")

    assert "performance" in result["answer"]
    assert "redundancy" in result["answer"]
    assert "[ECU-750-2, ECU-850-3]" in result["answer"]
    assert {citation["chunk_id"] for citation in result["citations"]} == {
        "ECU-750-2",
        "ECU-850-3",
    }


def test_no_evidence_returns_no_citations(agent):
    result = agent.invoke("What is the ECU-750 Bluetooth firmware version?")

    assert result["answer"] == (
        "No retrieved evidence supports a reliable answer to this question."
    )
    assert result["citations"] == []
    assert "[" not in result["answer"]


def test_paraphrased_processor_question_stays_grounded(agent):
    result = agent.invoke("What silicon and clock rate does the ECU-850b use?")

    assert result["routed_models"] == ["ECU-850b"]
    assert result["field"] == "processor"
    assert "Dual-core ARM Cortex-A53" in result["answer"]
    assert "1.5 GHz" in result["answer"]
    assert "[ECU-850b-2]" in result["answer"]
    assert {citation["chunk_id"] for citation in result["citations"]} == {
        "ECU-850b-2"
    }
    assert result["needs_human_review"] is False


def test_unknown_domain_question_is_not_mapped_to_capacity(agent):
    result = agent.invoke("What is the fuel tank capacity for ECU-850?")

    assert result["routed_models"] == ["ECU-850"]
    assert result["field"] is None
    assert result["answer"] == (
        "No retrieved evidence supports a reliable answer to this question."
    )
    assert result["citations"] == []
    assert result["needs_human_review"] is True


@pytest.mark.parametrize(
    "question",
    [
        "Can ECU-850 update itself wirelessly?",
        "Can the ECU-850 receive remote software upgrades without a cable?",
    ],
)
def test_single_model_ota_paraphrases_do_not_route_to_can(agent, question):
    result = agent.invoke(question)

    assert result["routed_models"] == ["ECU-850"]
    assert result["field"] == "ota"
    assert "ECU-850 supports OTA updates" in result["answer"]
    assert "CAN specification" not in result["answer"]
    assert {citation["chunk_id"] for citation in result["citations"]} == {
        "ECU-850-2"
    }
    assert {citation["section"] for citation in result["citations"]} == {"Key Features"}
    assert result["needs_human_review"] is False


def test_single_model_inherited_ota_keeps_explicit_scope(agent):
    result = agent.invoke("Can 850b receive remote software upgrades without a cable?")

    assert result["routed_models"] == ["ECU-850b"]
    assert result["field"] == "ota"
    assert "ECU-850b supports OTA updates" in result["answer"]
    assert "ECU-750" not in result["answer"]
    assert {citation["chunk_id"] for citation in result["citations"]} == {
        "ECU-850-2",
        "ECU-850b-1",
    }
    assert result["needs_human_review"] is False


def test_wrong_model_question_returns_no_evidence_instead_of_fleet_answer(agent):
    result = agent.invoke("Does the ECU-650 support OTA?")

    assert result["routed_models"] == []
    assert result["field"] == "ota"
    assert result["answer"] == (
        "No retrieved evidence supports a reliable answer to this question."
    )
    assert result["citations"] == []
    assert result["needs_human_review"] is True


def test_mixed_valid_and_wrong_model_question_returns_no_evidence(agent):
    result = agent.invoke("Compare storage for ECU-650 and ECU-850.")

    assert result["routed_models"] == []
    assert result["field"] == "storage"
    assert result["answer"] == (
        "No retrieved evidence supports a reliable answer to this question."
    )
    assert result["citations"] == []
    assert result["needs_human_review"] is True
