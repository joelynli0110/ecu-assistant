"""Routing, answer generation, and LangGraph node implementations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from ecu_assistant.agent.prompts import SYSTEM_PROMPT, build_chat_model
from ecu_assistant.agent.state import AgentState
from ecu_assistant.config import AgentConfig
from ecu_assistant.data.loaders import DocumentRepository, detect_spec_field
from ecu_assistant.data.schemas import (
    ALL_MODELS,
    AnswerResult,
    ModelRecord,
    RouteDecision,
)
from ecu_assistant.retrieval.retriever import ECURetriever


class QueryRouter:
    """Route explicit model mentions and infer broad comparison queries."""

    MODEL_PATTERNS = {
        "ECU-850b": re.compile(
            r"\b(?:ecu[-_\s]*)?850[-_\s]*b\b",
            re.IGNORECASE,
        ),
        "ECU-850": re.compile(
            r"\b(?:ecu[-_\s]*)?850(?![-_\s]*b\b)\b",
            re.IGNORECASE,
        ),
        "ECU-750": re.compile(r"\b(?:ecu[-_\s]*)?750\b", re.IGNORECASE),
    }

    def __init__(self, records: Mapping[str, ModelRecord] | None = None):
        self.records = records or {}

    @staticmethod
    def _intent(query: str) -> str:
        lower = query.lower()
        if any(
            term in lower
            for term in (
                "compare", "comparison", "contrast", "difference", "versus",
                " vs",
            )
        ):
            return "comparison"
        if any(term in lower for term in ("which", "all ecu", "across")):
            return "fleet_query"
        if any(term in lower for term in ("enable", "configure", "command")):
            return "configuration"
        return "specification"

    def route(self, query: str) -> RouteDecision:
        """Select documents using model mentions, series mentions, and intent."""

        models = [
            model for model, pattern in self.MODEL_PATTERNS.items() if pattern.search(query)
        ]
        lower = query.lower()
        if not models and ("ecu-700" in lower or "700 series" in lower):
            models.append("ECU-750")
        if not models and ("ecu-800" in lower or "800 series" in lower):
            models.extend(["ECU-850", "ECU-850b"])

        if not models:
            models = list(ALL_MODELS)
            reason = "No specific model was named; search the complete ECU corpus."
        else:
            models = [model for model in ALL_MODELS if model in models]
            reason = f"Explicit model or series references selected: {', '.join(models)}."
        return RouteDecision(
            models=models,
            intent=self._intent(query),
            field=detect_spec_field(query, self.records),
            reason=reason,
        )


def _source_list(records: list[ModelRecord]) -> str:
    return ", ".join(record.source for record in records)


class ExtractiveAnswerer:
    """Answer common engineering questions from parsed source facts."""

    def __init__(self, repository: DocumentRepository):
        self.repository = repository

    @staticmethod
    def _spec(record: ModelRecord, name: str) -> str:
        return record.specs.get(name, "not documented")

    @staticmethod
    def _command(record: ModelRecord) -> str | None:
        match = re.search(r"me-driver-ctl\s+--enable-npu\s+--mode=performance", record.text)
        return match.group(0) if match else None

    def _comparison(self, query: str, records: list[ModelRecord]) -> AnswerResult | None:
        lower = query.lower()
        by_model = {record.model: record for record in records}
        if not (("compare" in lower or "difference" in lower) and len(records) >= 2):
            return None
        if {record.model for record in records} == {"ECU-850", "ECU-850b"}:
            base, enhanced = by_model["ECU-850"], by_model["ECU-850b"]
            return AnswerResult(
                "The ECU-850b upgrades the ECU-850 with a 5 TOPS dedicated NPU, "
                f"{self._spec(enhanced, 'memory')} RAM instead of "
                f"{self._spec(base, 'memory')}, and a "
                f"{self._spec(enhanced, 'processor')} instead of "
                f"{self._spec(base, 'processor')}. It also increases storage from "
                f"{self._spec(base, 'storage')} to {self._spec(enhanced, 'storage')}. "
                f"[Sources: {_source_list(records)}]",
                0.98,
            )
        if "can" in lower:
            details = "; ".join(
                f"{record.model}: {self._spec(record, 'can')}" for record in records
            )
            return AnswerResult(
                f"{details}. The newer dual-channel 2 Mbps interface provides more "
                f"bandwidth and channel redundancy. [Sources: {_source_list(records)}]",
                0.97,
            )
        return None

    def _generic_spec_answer(
        self,
        query: str,
        records: list[ModelRecord],
        field: str | None = None,
    ) -> AnswerResult | None:
        """Answer any field parsed from source tables or feature metadata."""

        field = field or detect_spec_field(query, self.repository.records)
        if not field or not records:
            return None

        models = [record.model for record in records]
        values = self.repository.compare_specs(models, field)
        documented = {model: value for model, value in values.items() if value is not None}
        if not documented:
            return None

        if field == "ota":
            supported = [model for model, value in documented.items() if value == "Supported"]
            unsupported = [
                model for model, value in documented.items() if value == "Not supported"
            ]
            parts = []
            if supported:
                parts.append(f"OTA updates are supported by {', '.join(supported)}")
            if unsupported:
                parts.append(f"{', '.join(unsupported)} does not support OTA updates")
            return AnswerResult(
                f"{'. '.join(parts)}. [Sources: {_source_list(records)}]",
                0.98,
            )

        label = field.upper() if field in {"can", "npu"} else field
        if len(records) == 1:
            record = records[0]
            value = documented.get(record.model)
            if value is None:
                return None
            if field == "npu":
                return AnswerResult(
                    f"The {record.model} features a dedicated Neural Processing Unit "
                    f"(NPU) capable of {value.replace(' AI Accelerator', '')} "
                    "(Tera Operations Per Second) for AI acceleration, making it "
                    f"suitable for edge AI workloads. [Source: {record.source}]",
                    0.99,
                )
            return AnswerResult(
                f"The {record.model} {label} specification is {value}. "
                f"[Source: {record.source}]",
                0.99,
            )

        details = "; ".join(
            f"{model}: {value if value is not None else 'not documented'}"
            for model, value in values.items()
        )
        implication = (
            " Higher channel count and bitrate provide more bandwidth and redundancy."
            if field == "can"
            else ""
        )
        return AnswerResult(
            f"{label.capitalize()} comparison: {details}.{implication} "
            f"[Sources: {_source_list(records)}]",
            0.98,
        )

    def _fleet_answer(self, query: str, records: list[ModelRecord]) -> AnswerResult | None:
        lower = query.lower()
        if "which" in lower and ("temperature" in lower or "harshest" in lower):
            maxima: dict[str, int] = {}
            for record in records:
                match = re.search(
                    r"\+(\d+)°C",
                    self._spec(record, "operating temperature"),
                )
                if match:
                    maxima[record.model] = int(match.group(1))
            if maxima:
                best = max(maxima.values())
                winners = [model for model, maximum in maxima.items() if maximum == best]
                others = "; ".join(
                    (
                        f"{model}: "
                        f"{self._spec(self.repository.records[model], 'operating temperature')}"
                    )
                    for model in maxima
                    if model not in winners
                )
                return AnswerResult(
                    f"{' and '.join(winners)} can operate in the harshest conditions: "
                    f"-40°C to +{best}°C. {others}. [Sources: {_source_list(records)}]",
                    0.97,
                )
        return None

    def _single_answer(self, query: str, record: ModelRecord) -> AnswerResult | None:
        lower = query.lower()
        if ("enable" in lower or "command" in lower or "configure" in lower) and "npu" in lower:
            command = self._command(record)
            if command:
                return AnswerResult(
                    f"To enable the NPU on the ECU-850b, run `{command}`. "
                    f"[Source: {record.source}]",
                    0.99,
                )
        return None

    def answer(
        self,
        query: str,
        models: list[str],
        field: str | None = None,
    ) -> AnswerResult:
        """Generate a deterministic document-derived answer."""

        records = self.repository.records_for(models)
        result = self._fleet_answer(query, records)
        if not result and len(records) == 1:
            result = self._single_answer(query, records[0])
        result = result or self._generic_spec_answer(query, records, field)
        result = result or self._comparison(query, records)
        if result:
            return result
        context = " ".join(record.text[:300] for record in records)
        return AnswerResult(
            "I found potentially relevant documentation, but could not extract a reliable "
            f"answer for this question. Available context: {context}",
            0.35,
            needs_human_review=True,
        )


class GroundedLLMAnswerer:
    """Provider-neutral grounded generation through a LangChain chat model."""

    def __init__(self, config: AgentConfig, model=None):
        self.model = model or build_chat_model(config)

    def answer(
        self,
        query: str,
        documents: list[Document],
        route_reason: str,
    ) -> AnswerResult:
        """Invoke the model with retrieved context and strict grounding rules."""

        context = "\n\n".join(
            (
                f"SOURCE={doc.metadata['source']} "
                f"MODEL={doc.metadata['model']} SECTION={doc.metadata['section']}\n"
                f"{doc.page_content}"
            )
            for doc in documents
        )
        prompt = (
            f"Routing rationale: {route_reason}\n\n"
            f"Retrieved documentation:\n{context}\n\n"
            f"Engineer question: {query}"
        )
        response = self.model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        text = str(response.content).strip()
        grounded = bool(text) and any(
            doc.metadata["source"].lower() in text.lower() for doc in documents
        )
        return AnswerResult(
            text=text,
            confidence=0.86 if grounded else 0.58,
            needs_human_review=not grounded,
        )


class AgentNodes:
    """Bound LangGraph node functions and confidence-routing policy."""

    def __init__(
        self,
        config: AgentConfig,
        repository: DocumentRepository,
        retriever: ECURetriever,
    ):
        self.config = config
        self.router = QueryRouter(repository.records)
        self.retriever = retriever
        self.extractive_answerer = ExtractiveAnswerer(repository)
        self.llm_answerer = (
            None
            if config.llm_provider in {"none", "extractive", "local"}
            else GroundedLLMAnswerer(config)
        )

    def route_query(self, state: AgentState) -> dict[str, Any]:
        """Route a question to ECU models and an intent."""

        decision = self.router.route(state["query"])
        return {
            "routed_models": decision.models,
            "intent": decision.intent,
            "field": decision.field,
            "route_reason": decision.reason,
            "retrieval_attempt": 0,
            "broaden": False,
        }

    def retrieve_context(self, state: AgentState) -> dict[str, Any]:
        """Retrieve chunks for the routed models."""

        return {
            "documents": self.retriever.search(
                state["query"],
                state["routed_models"],
                broaden=state.get("broaden", False),
            )
        }

    def generate_answer(self, state: AgentState) -> dict[str, Any]:
        """Generate an answer and collect source citations."""

        if self.llm_answerer:
            result = self.llm_answerer.answer(
                state["query"], state["documents"], state["route_reason"]
            )
        else:
            result = self.extractive_answerer.answer(
                state["query"],
                state["routed_models"],
                state.get("field"),
            )
        citations = sorted(
            {
                document.metadata["source"]
                for document in state["documents"]
                if document.metadata["source"].lower() in result.text.lower()
            }
        ) or sorted({document.metadata["source"] for document in state["documents"]})
        return {
            "answer": result.text,
            "confidence": result.confidence,
            "needs_human_review": result.needs_human_review,
            "citations": citations,
        }

    def assess_confidence(self, state: AgentState) -> dict[str, Any]:
        """Apply confidence and human-review policy."""

        answer = state.get("answer", "")
        confidence = state.get("confidence", 0.0)
        if not re.search(r"\d|supported|not support|document", answer, re.I):
            confidence = min(confidence, 0.4)
        return {
            "confidence": confidence,
            "needs_human_review": (
                confidence < self.config.low_confidence_threshold
                or state.get("needs_human_review", False)
            ),
        }

    def after_assessment(self, state: AgentState) -> str:
        """Choose whether to broaden chunk search within the routed scope."""

        if (
            state.get("confidence", 0.0) < self.config.low_confidence_threshold
            and state.get("retrieval_attempt", 0) == 0
        ):
            return "broaden"
        return "finish"

    @staticmethod
    def broaden_retrieval(state: AgentState) -> dict[str, Any]:
        """Broaden chunk search without changing the routed model scope."""

        return {
            "route_reason": (
                f"{state['route_reason']} Low confidence triggered a broader "
                "chunk search within the same model scope."
            ),
            "retrieval_attempt": state.get("retrieval_attempt", 0) + 1,
            "broaden": True,
        }
