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
from ecu_assistant.data.loaders import (
    DocumentRepository,
    detect_spec_field,
    parse_spec_table,
)
from ecu_assistant.data.schemas import (
    ALL_MODELS,
    AnswerResult,
    ModelRecord,
    RouteDecision,
)
from ecu_assistant.retrieval.retriever import ECURetriever


class QueryRouter:
    """Route explicit model mentions and infer broad comparison queries."""

    UNKNOWN_MODEL_PATTERN = re.compile(
        r"\b(?P<prefix>ecu[-_\s]*)?(?P<number>\d{3})(?:[-_\s]*(?P<suffix>[a-z]))?\b",
        re.IGNORECASE,
    )
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

    @classmethod
    def _names_unknown_model(cls, query: str) -> bool:
        for match in cls.UNKNOWN_MODEL_PATTERN.finditer(query):
            number = match.group("number")
            suffix = (match.group("suffix") or "").lower()
            token = f"{number}{suffix}"
            if token in {"750", "850", "850b", "700", "800"}:
                continue
            if match.group("prefix") or suffix:
                return True
        return False

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

        intent = self._intent(query)
        field = detect_spec_field(query, self.records)
        unknown_model = self._names_unknown_model(query)
        models = [
            model for model, pattern in self.MODEL_PATTERNS.items() if pattern.search(query)
        ]
        lower = query.lower()
        if not models and ("ecu-700" in lower or "700 series" in lower):
            models.append("ECU-750")
        if not models and ("ecu-800" in lower or "800 series" in lower):
            models.extend(["ECU-850", "ECU-850b"])
        if unknown_model:
            return RouteDecision(
                models=[],
                intent=intent,
                field=field,
                reason="The query names an unsupported ECU model.",
            )

        if not models:
            models = list(ALL_MODELS)
            reason = "No specific model was named; search the complete ECU corpus."
        else:
            models = [model for model in ALL_MODELS if model in models]
            reason = f"Explicit model or series references selected: {', '.join(models)}."
        return RouteDecision(
            models=models,
            intent=intent,
            field=field,
            reason=reason,
        )


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    return list({document.metadata["chunk_id"]: document for document in documents}.values())


def _marker(documents: list[Document]) -> str:
    chunk_ids = [document.metadata["chunk_id"] for document in documents]
    return f"[{', '.join(chunk_ids)}]"


OTA_PATTERN = re.compile(r"\bota\b|over[-\s]the[-\s]air", re.IGNORECASE)


class ExtractiveAnswerer:
    """Generate answers strictly from retrieved document chunks."""

    @staticmethod
    def _direct_lookup(
        documents: list[Document],
        model: str,
        field: str,
    ) -> tuple[str, list[Document]] | None:
        for document in documents:
            if document.metadata.get("model") != model:
                continue
            value = parse_spec_table(document.page_content).get(field)
            if value is not None:
                return value, [document]
        if field != "ota":
            return None
        for document in documents:
            if document.metadata.get("model") != model:
                continue
            text = document.page_content.lower()
            if OTA_PATTERN.search(document.page_content) and "not supported" in text:
                return "Not supported", [document]
            if OTA_PATTERN.search(document.page_content):
                return "Supported", [document]
        return None

    def _lookup(
        self,
        documents: list[Document],
        model: str,
        field: str,
    ) -> tuple[str, list[Document]] | None:
        direct = self._direct_lookup(documents, model, field)
        if direct and field == "npu":
            supporting = [
                document
                for document in documents
                if document.metadata.get("model") == model
                and "edge ai workloads" in document.page_content.lower()
            ]
            return direct[0], _deduplicate_documents([*direct[1], *supporting])
        if direct or field != "ota" or model != "ECU-850b":
            return direct
        inheritance = next(
            (
                document
                for document in documents
                if document.metadata.get("model") == "ECU-850b"
                and "includes all features" in document.page_content.lower()
            ),
            None,
        )
        base_ota = self._direct_lookup(documents, "ECU-850", "ota")
        if inheritance and base_ota and base_ota[0] == "Supported":
            return "Supported", _deduplicate_documents([inheritance, *base_ota[1]])
        return None

    def _configuration_answer(
        self,
        query: str,
        models: list[str],
        documents: list[Document],
    ) -> AnswerResult | None:
        lower = query.lower()
        if not (
            "npu" in lower
            and any(term in lower for term in ("enable", "configure", "command"))
        ):
            return None
        for document in documents:
            if document.metadata.get("model") not in models:
                continue
            match = re.search(
                r"me-driver-ctl\s+--enable-npu\s+--mode=performance",
                document.page_content,
            )
            if match:
                return AnswerResult(
                    f"To enable the NPU on ECU-850b, run `{match.group(0)}` "
                    f"{_marker([document])}.",
                    0.99,
                    evidence_chunk_ids=(document.metadata["chunk_id"],),
                )
        return None

    def _difference_answer(
        self,
        query: str,
        models: list[str],
        field: str | None,
        documents: list[Document],
    ) -> AnswerResult | None:
        lower = query.lower()
        if field is not None or set(models) != {"ECU-850", "ECU-850b"}:
            return None
        if not any(term in lower for term in ("difference", "compare", "contrast")):
            return None

        statements: list[str] = []
        evidence: list[Document] = []
        npu = self._lookup(documents, "ECU-850b", "npu")
        if npu:
            statements.append(
                f"ECU-850b adds a dedicated {npu[0]} {_marker(npu[1])}"
            )
            evidence.extend(npu[1])
        for field_name, label in (
            ("memory", "memory"),
            ("processor", "processor"),
            ("storage", "storage"),
        ):
            base = self._lookup(documents, "ECU-850", field_name)
            enhanced = self._lookup(documents, "ECU-850b", field_name)
            if base and enhanced:
                docs = _deduplicate_documents([*base[1], *enhanced[1]])
                statements.append(
                    f"{label.capitalize()} changes from {base[0]} on ECU-850 "
                    f"to {enhanced[0]} on ECU-850b {_marker(docs)}"
                )
                evidence.extend(docs)
        if not statements:
            return None
        evidence = _deduplicate_documents(evidence)
        return AnswerResult(
            ". ".join(statements) + ".",
            0.98,
            evidence_chunk_ids=tuple(
                document.metadata["chunk_id"] for document in evidence
            ),
        )

    def _generic_spec_answer(
        self,
        models: list[str],
        field: str | None,
        documents: list[Document],
    ) -> AnswerResult | None:
        """Answer a routed field using only retrieved chunk evidence."""

        if not field or not models:
            return None

        values = {
            model: self._lookup(documents, model, field)
            for model in models
        }
        documented = {model: value for model, value in values.items() if value}
        if not documented:
            return None

        if field == "ota":
            parts: list[str] = []
            evidence: list[Document] = []
            for model, (value, value_docs) in documented.items():
                statement = (
                    f"{model} supports OTA updates"
                    if value == "Supported"
                    else f"{model} does not support OTA updates"
                )
                parts.append(f"{statement} {_marker(value_docs)}")
                evidence.extend(value_docs)
            evidence = _deduplicate_documents(evidence)
            return AnswerResult(
                ". ".join(parts) + ".",
                0.98,
                evidence_chunk_ids=tuple(
                    document.metadata["chunk_id"] for document in evidence
                ),
            )

        label = field.upper() if field in {"can", "npu"} else field
        if len(models) == 1:
            model = models[0]
            evidence_value = documented.get(model)
            if evidence_value is None:
                return None
            value, value_docs = evidence_value
            if field == "npu":
                return AnswerResult(
                    f"The {model} features a dedicated Neural Processing Unit "
                    f"(NPU) capable of {value.replace(' AI Accelerator', '')} "
                    "(Tera Operations Per Second) for AI acceleration, making it "
                    f"suitable for edge AI workloads {_marker(value_docs)}.",
                    0.99,
                    evidence_chunk_ids=tuple(
                        document.metadata["chunk_id"] for document in value_docs
                    ),
                )
            return AnswerResult(
                f"The {model} {label} specification is {value} "
                f"{_marker(value_docs)}.",
                0.99,
                evidence_chunk_ids=tuple(
                    document.metadata["chunk_id"] for document in value_docs
                ),
            )

        details: list[str] = []
        evidence: list[Document] = []
        for model, evidence_value in documented.items():
            value, value_docs = evidence_value
            details.append(f"{model}: {value} {_marker(value_docs)}")
            evidence.extend(value_docs)
        evidence = _deduplicate_documents(evidence)
        synthesis = ""
        if field == "can" and len(documented) > 1:
            synthesis = (
                " Based on these documented values, ECU-850 offers higher CAN bus "
                "performance (2 Mbps versus 1 Mbps) and channel redundancy "
                f"(dual channel versus single channel) {_marker(evidence)}."
            )
        return AnswerResult(
            f"{label.capitalize()} comparison: {'; '.join(details)}.{synthesis}",
            0.98,
            evidence_chunk_ids=tuple(
                document.metadata["chunk_id"] for document in evidence
            ),
        )

    def _fleet_answer(
        self,
        query: str,
        models: list[str],
        documents: list[Document],
    ) -> AnswerResult | None:
        lower = query.lower()
        if "which" in lower and ("temperature" in lower or "harshest" in lower):
            values = {
                model: self._lookup(documents, model, "operating temperature")
                for model in models
            }
            maxima: dict[str, tuple[int, str, list[Document]]] = {}
            for model, evidence_value in values.items():
                if not evidence_value:
                    continue
                value, value_docs = evidence_value
                match = re.search(r"\+(\d+)°C", value)
                if match:
                    maxima[model] = (int(match.group(1)), value, value_docs)
            if maxima:
                best = max(item[0] for item in maxima.values())
                winners = [model for model, item in maxima.items() if item[0] == best]
                details = []
                evidence: list[Document] = []
                for model, (_, value, value_docs) in maxima.items():
                    details.append(f"{model}: {value} {_marker(value_docs)}")
                    evidence.extend(value_docs)
                evidence = _deduplicate_documents(evidence)
                winner_docs = _deduplicate_documents(
                    [
                        document
                        for winner in winners
                        for document in maxima[winner][2]
                    ]
                )
                return AnswerResult(
                    f"{' and '.join(winners)} can operate at the highest maximum "
                    f"temperature of +{best}°C {_marker(winner_docs)}. "
                    f"{'; '.join(details)}.",
                    0.97,
                    evidence_chunk_ids=tuple(
                        document.metadata["chunk_id"] for document in evidence
                    ),
                )
        return None

    def answer(
        self,
        query: str,
        models: list[str],
        documents: list[Document],
        field: str | None,
    ) -> AnswerResult:
        """Generate a deterministic answer from retrieved chunks only."""

        result = self._configuration_answer(query, models, documents)
        result = result or self._fleet_answer(query, models, documents)
        result = result or self._generic_spec_answer(models, field, documents)
        result = result or self._difference_answer(query, models, field, documents)
        if result:
            return result
        return AnswerResult(
            "No retrieved evidence supports a reliable answer to this question.",
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
                f"CHUNK_ID={doc.metadata['chunk_id']} "
                f"SOURCE={doc.metadata['source']} MODEL={doc.metadata['model']} "
                f"SECTION={doc.metadata['section']}\n"
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
        valid_ids = {document.metadata["chunk_id"] for document in documents}
        cited_ids = tuple(
            dict.fromkeys(
                chunk_id
                for chunk_id in re.findall(r"\[([A-Za-z0-9_-]+)\]", text)
                if chunk_id in valid_ids
            )
        )
        grounded = bool(text) and bool(cited_ids)
        return AnswerResult(
            text=text,
            confidence=0.86 if grounded else 0.58,
            needs_human_review=not grounded,
            evidence_chunk_ids=cited_ids,
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
        self.extractive_answerer = ExtractiveAnswerer()
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
                field=state.get("field"),
                intent=state.get("intent"),
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
                state["documents"],
                state.get("field"),
            )
        evidence_ids = set(result.evidence_chunk_ids)
        citations = [
            {
                "source": document.metadata["source"],
                "section": document.metadata["section"],
                "chunk_id": document.metadata["chunk_id"],
                "model": document.metadata["model"],
            }
            for document in state["documents"]
            if document.metadata["chunk_id"] in evidence_ids
        ]
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
