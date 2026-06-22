"""State passed between LangGraph nodes."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """Serializable state flowing through the assistant graph."""

    query: str
    routed_models: list[str]
    intent: str
    route_reason: str
    documents: list[Document]
    answer: str
    confidence: float
    citations: list[str]
    needs_human_review: bool
    retrieval_attempt: int
    broaden: bool

