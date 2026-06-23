"""Grounding prompt and provider-neutral chat-model construction."""

from __future__ import annotations

from ecu_assistant.config import AgentConfig

SYSTEM_PROMPT = """You are the ME Engineering Assistant.
Answer only from the supplied ECU documentation.
Rules:
1. Preserve exact model names, units, ranges, and shell commands.
2. For comparisons, state each model's value and the engineering implication.
3. Use only facts present in the retrieved chunks.
4. Cite every factual claim inline using its exact [chunk_id].
5. Never cite a chunk that does not support the associated claim.
6. If evidence is missing or ambiguous, say so and request engineering review.
7. Answer in the same language as the user.
"""


def build_chat_model(config: AgentConfig):
    """Create a LangChain chat model for the selected public provider."""

    provider = config.llm_provider
    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional provider
            raise RuntimeError("OpenAI generation requires `pip install -e .[openai]`.") from exc
        return ChatOpenAI(
            model=config.llm_model or "gpt-4o-mini",
            temperature=config.temperature,
            base_url=config.llm_base_url,
        )
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - optional provider
            raise RuntimeError(
                "Anthropic generation requires `pip install -e .[anthropic]`."
            ) from exc
        return ChatAnthropic(
            model=config.llm_model or "claude-3-5-haiku-latest",
            temperature=config.temperature,
        )
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional provider
            raise RuntimeError("Ollama generation requires `pip install -e .[ollama]`.") from exc
        options = {
            "model": config.llm_model or "llama3.2",
            "temperature": config.temperature,
        }
        if config.llm_base_url:
            options["base_url"] = config.llm_base_url
        return ChatOllama(**options)
    raise ValueError(
        f"Unsupported LLM provider '{provider}'. Choose none, openai, anthropic, or ollama."
    )
