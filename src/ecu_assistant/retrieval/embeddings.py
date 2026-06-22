"""Local and public-provider embedding adapters."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

from ecu_assistant.config import AgentConfig


class LocalHashEmbeddings(Embeddings):
    """Dependency-light deterministic embeddings for offline execution."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9.+-]+|[\u4e00-\u9fff]", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "little") % self.dimensions
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(config: AgentConfig) -> Embeddings:
    """Build the configured LangChain embedding adapter."""

    provider = config.embedding_provider
    if provider in {"hash", "local", "none"}:
        return LocalHashEmbeddings()
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - optional provider
            raise RuntimeError("OpenAI embeddings require `pip install -e .[openai]`.") from exc
        return OpenAIEmbeddings(
            model=config.embedding_model or "text-embedding-3-small",
            base_url=config.embedding_base_url,
        )
    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover - optional provider
            raise RuntimeError("Ollama embeddings require `pip install -e .[ollama]`.") from exc
        options = {"model": config.embedding_model or "nomic-embed-text"}
        if config.embedding_base_url:
            options["base_url"] = config.embedding_base_url
        return OllamaEmbeddings(**options)
    raise ValueError(
        f"Unsupported embedding provider '{provider}'. Choose hash, openai, or ollama."
    )

