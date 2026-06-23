"""Model-aware retrieval over the in-memory vector store."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ecu_assistant.config import AgentConfig
from ecu_assistant.data.loaders import parse_spec_table
from ecu_assistant.retrieval.embeddings import build_embeddings
from ecu_assistant.retrieval.vector_store import build_vector_store


class ECURetriever:
    """Search chunks and constrain results to router-selected models."""

    def __init__(
        self,
        documents: Iterable[Document],
        config: AgentConfig,
        embeddings: Embeddings | None = None,
    ):
        self.documents = list(documents)
        self.config = config
        self.vector_store = build_vector_store(
            self.documents,
            embeddings or build_embeddings(config),
        )

    @staticmethod
    def _supports_request(
        document: Document,
        field: str | None,
        intent: str | None,
    ) -> bool:
        """Return whether a chunk contains direct evidence for the routed request."""

        text = document.page_content.lower()
        if field and field in parse_spec_table(document.page_content):
            return True
        if field == "ota":
            return (
                "over-the-air" in text
                or "ota" in text
                or "includes all features" in text
            )
        if intent == "configuration":
            return "enable" in text and "npu" in text
        if intent == "comparison" and field is None:
            return (
                "technical specifications" in text
                or "key differentiators" in text
                or "key upgrades" in text
            )
        return False

    def search(
        self,
        query: str,
        models: list[str],
        field: str | None = None,
        intent: str | None = None,
        broaden: bool = False,
    ) -> list[Document]:
        """Return relevant chunks, preserving evidence for every routed model."""

        allowed = set(models)
        ranked = self.vector_store.similarity_search(query, k=len(self.documents))
        selected = [doc for doc in ranked if doc.metadata.get("model") in allowed]
        selected.sort(
            key=lambda document: self._supports_request(document, field, intent),
            reverse=True,
        )

        result_k = len(selected) if broaden else self.config.retrieval_k
        present = {doc.metadata.get("model") for doc in selected[:result_k]}
        result = selected[:result_k]
        for model in models:
            if model not in present:
                candidate = next(
                    (doc for doc in selected if doc.metadata.get("model") == model),
                    None,
                )
                if candidate:
                    result.append(candidate)

        if field == "ota" and "ECU-850b" in models:
            inheritance = next(
                (
                    doc
                    for doc in self.documents
                    if doc.metadata.get("model") == "ECU-850b"
                    and "includes all features" in doc.page_content.lower()
                ),
                None,
            )
            base_ota = next(
                (
                    doc
                    for doc in self.documents
                    if doc.metadata.get("model") == "ECU-850"
                    and (
                        "over-the-air" in doc.page_content.lower()
                        or "ota" in doc.page_content.lower()
                    )
                ),
                None,
            )
            result.extend(doc for doc in (inheritance, base_ota) if doc is not None)

        return list({doc.metadata["chunk_id"]: doc for doc in result}.values())
