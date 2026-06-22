"""End-to-end tests against the challenge questions."""

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

    assert result["routed_models"] == ["ECU-750", "ECU-850", "ECU-850b"]
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
                "The ECU-850 has 2 GB LPDDR4 RAM. "
                "[Source: ECU-800_Series_Base.md, ECU-850 Technical Specifications]"
            ]
        ),
    )

    result = answerer.answer("How much RAM does the ECU-850 have?", documents, "test")

    assert result.confidence == 0.86
    assert result.needs_human_review is False

