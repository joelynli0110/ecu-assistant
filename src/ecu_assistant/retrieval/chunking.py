"""Heading-aware chunking for ECU Markdown documents."""

from __future__ import annotations

import re

from langchain_core.documents import Document

from ecu_assistant.data.schemas import ModelRecord


def chunk_records(records: dict[str, ModelRecord]) -> list[Document]:
    """Split records by Markdown heading while preserving traceable metadata."""

    chunks: list[Document] = []
    for model, record in records.items():
        sections = re.split(r"(?=^#{1,3}\s)", record.text, flags=re.MULTILINE)
        for index, section in enumerate(part.strip() for part in sections if part.strip()):
            heading_match = re.match(r"^#{1,3}\s+(.+)", section)
            heading = heading_match.group(1).strip() if heading_match else "Document"
            chunks.append(
                Document(
                    page_content=section,
                    metadata={
                        "model": model,
                        "series": record.series,
                        "source": record.source,
                        "section": heading,
                        "chunk_id": f"{model}-{index}",
                    },
                )
            )
    return chunks

