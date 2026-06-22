"""Data loading, chunking, embedding, and retrieval tests."""

from ecu_assistant.config import AgentConfig
from ecu_assistant.data.loaders import DocumentRepository
from ecu_assistant.retrieval.chunking import chunk_records
from ecu_assistant.retrieval.embeddings import LocalHashEmbeddings, build_embeddings
from ecu_assistant.retrieval.retriever import ECURetriever


def test_repository_extracts_normalized_facts():
    repository = DocumentRepository()

    assert set(repository.records) == {"ECU-750", "ECU-850", "ECU-850b"}
    assert repository.records["ECU-750"].specs["operating temperature"] == "-40°C to +85°C"
    assert repository.records["ECU-850"].specs["memory"] == "2 GB LPDDR4"
    assert repository.records["ECU-850b"].ota_supported is True


def test_chunks_are_traceable():
    chunks = chunk_records(DocumentRepository().records)

    assert chunks
    assert all(
        {"model", "series", "source", "section", "chunk_id"} <= document.metadata.keys()
        for document in chunks
    )


def test_default_embeddings_are_local_and_credential_free(monkeypatch):
    monkeypatch.delenv("ME_ECU_EMBEDDING_PROVIDER", raising=False)
    config = AgentConfig.from_env()

    assert isinstance(build_embeddings(config), LocalHashEmbeddings)


def test_retriever_preserves_both_comparison_models():
    repository = DocumentRepository()
    retriever = ECURetriever(
        chunk_records(repository.records),
        AgentConfig(),
    )

    documents = retriever.search(
        "Compare the CAN bus capabilities.",
        ["ECU-750", "ECU-850"],
    )

    assert {"ECU-750", "ECU-850"} <= {
        document.metadata["model"] for document in documents
    }

