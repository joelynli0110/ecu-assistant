"""LangGraph assembly and public agent facade."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ecu_assistant.agent.nodes import AgentNodes
from ecu_assistant.agent.state import AgentState
from ecu_assistant.config import AgentConfig
from ecu_assistant.data.loaders import DocumentRepository
from ecu_assistant.retrieval.chunking import chunk_records
from ecu_assistant.retrieval.retriever import ECURetriever


class ECUEngineeringAgent:
    """Production-facing LangGraph agent with local fallback behavior."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.repository = DocumentRepository(self.config.docs_dir)
        documents = chunk_records(self.repository.records)
        self.retriever = ECURetriever(documents, self.config)
        self.nodes = AgentNodes(self.config, self.repository, self.retriever)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("route_query", self.nodes.route_query)
        workflow.add_node("retrieve_context", self.nodes.retrieve_context)
        workflow.add_node("generate_answer", self.nodes.generate_answer)
        workflow.add_node("assess_confidence", self.nodes.assess_confidence)
        workflow.add_node("broaden_retrieval", self.nodes.broaden_retrieval)
        workflow.add_edge(START, "route_query")
        workflow.add_edge("route_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "assess_confidence")
        workflow.add_conditional_edges(
            "assess_confidence",
            self.nodes.after_assessment,
            {"broaden": "broaden_retrieval", "finish": END},
        )
        workflow.add_edge("broaden_retrieval", "retrieve_context")
        return workflow.compile()

    def invoke(self, query: str) -> dict[str, Any]:
        """Run a single question and return a serving-friendly response."""

        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "A non-empty engineering question is required.",
                "confidence": 0.0,
                "citations": [],
                "routed_models": [],
                "intent": "invalid",
                "field": None,
                "needs_human_review": True,
                "review_reason": "empty_query",
            }
        state = self.graph.invoke({"query": clean_query})
        return {
            "answer": state["answer"],
            "confidence": round(float(state["confidence"]), 3),
            "citations": state["citations"],
            "routed_models": state["routed_models"],
            "intent": state["intent"],
            "field": state.get("field"),
            "needs_human_review": state["needs_human_review"],
            "review_reason": state.get("review_reason", "grounded"),
        }
