"""Model-aware retrieval over the in-memory vector store."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ecu_assistant.config import AgentConfig
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

    def search(self, query: str, models: list[str], broaden: bool = False) -> list[Document]:
        """Return relevant chunks, preserving evidence for every routed model."""

        allowed = set(models)
        search_k = len(self.documents) if broaden else min(
            len(self.documents), max(self.config.retrieval_k * 3, 8)
        )
        ranked = self.vector_store.similarity_search(query, k=search_k)
        selected = [doc for doc in ranked if doc.metadata.get("model") in allowed]

        present = {doc.metadata.get("model") for doc in selected[: self.config.retrieval_k]}
        result = selected[: self.config.retrieval_k]
        for model in models:
            if model not in present:
                candidate = next(
                    (doc for doc in selected if doc.metadata.get("model") == model),
                    None,
                )
                if candidate:
                    result.append(candidate)

        return list({doc.metadata["chunk_id"]: doc for doc in result}.values())

