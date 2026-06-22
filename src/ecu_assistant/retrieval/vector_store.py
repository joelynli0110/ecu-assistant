"""In-memory vector-store construction."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_vector_store(
    documents: Iterable[Document],
    embeddings: Embeddings,
) -> InMemoryVectorStore:
    """Index documents in a LangChain in-memory vector store."""

    store = InMemoryVectorStore(embeddings)
    store.add_documents(list(documents))
    return store

