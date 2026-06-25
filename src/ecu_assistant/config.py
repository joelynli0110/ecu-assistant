"""Environment-driven runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    """Configuration shared by retrieval, generation, and serving."""

    docs_dir: Path | None = None
    llm_provider: str = "none"
    llm_model: str | None = None
    llm_base_url: str | None = None
    embedding_provider: str = "hash"
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    retrieval_k: int = 4
    temperature: float = 0.0
    low_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.90

    @classmethod
    def from_env(cls, docs_dir: str | Path | None = None) -> "AgentConfig":
        """Build configuration from environment variables."""

        configured_docs = docs_dir or os.getenv("ME_ECU_DOCS_DIR")
        return cls(
            docs_dir=Path(configured_docs) if configured_docs else None,
            llm_provider=os.getenv("ME_ECU_LLM_PROVIDER", "none").lower(),
            llm_model=os.getenv("ME_ECU_LLM_MODEL"),
            llm_base_url=os.getenv("ME_ECU_LLM_BASE_URL"),
            embedding_provider=os.getenv("ME_ECU_EMBEDDING_PROVIDER", "hash").lower(),
            embedding_model=os.getenv("ME_ECU_EMBEDDING_MODEL"),
            embedding_base_url=os.getenv("ME_ECU_EMBEDDING_BASE_URL"),
            retrieval_k=int(os.getenv("ME_ECU_RETRIEVAL_K", "4")),
            temperature=float(os.getenv("ME_ECU_TEMPERATURE", "0")),
            low_confidence_threshold=float(
                os.getenv("ME_ECU_LOW_CONFIDENCE_THRESHOLD", "0.55")
            ),
            high_confidence_threshold=float(
                os.getenv("ME_ECU_HIGH_CONFIDENCE_THRESHOLD", "0.90")
            ),
        )
