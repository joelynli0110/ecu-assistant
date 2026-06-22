"""Load, normalize, and parse ECU Markdown documents."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from ecu_assistant.data.schemas import ModelRecord, SOURCE_FILES

FEATURE_ALIASES = {
    "operating temp.": "operating temperature",
    "memory (ram)": "memory",
    "can interface": "can",
    "power consumption": "power",
}


def normalize_text(text: str) -> str:
    """Repair known source encoding issues and normalize line endings."""

    return text.replace("掳C", "°C").replace("\r\n", "\n").strip()


def _clean_markdown(value: str) -> str:
    return re.sub(r"[*_`]", "", value).strip()


class DocumentRepository:
    """Load packaged or externally supplied ECU Markdown documents."""

    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = docs_dir
        self._texts = self._load_texts()
        self.records = self._parse_records()

    def _load_texts(self) -> dict[str, str]:
        texts: dict[str, str] = {}
        for model, filename in SOURCE_FILES.items():
            if self.docs_dir:
                path = self.docs_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"Missing ECU source document: {path}")
                raw = path.read_text(encoding="utf-8")
            else:
                raw = (
                    resources.files("ecu_assistant.data")
                    .joinpath("documents", filename)
                    .read_text(encoding="utf-8")
                )
            texts[model] = normalize_text(raw)
        return texts

    @staticmethod
    def _parse_table(text: str) -> dict[str, str]:
        specs: dict[str, str] = {}
        for raw_line in text.splitlines():
            if "|" not in raw_line or "---" in raw_line:
                continue
            parts = [_clean_markdown(part) for part in raw_line.strip().strip("|").split("|")]
            if len(parts) < 2:
                continue
            feature, value = parts[0].lower(), parts[1]
            feature = FEATURE_ALIASES.get(feature, feature)
            if feature not in {"feature", ""} and value.lower() != "specification":
                specs[feature] = value
        return specs

    def _parse_records(self) -> dict[str, ModelRecord]:
        records: dict[str, ModelRecord] = {}
        for model, text in self._texts.items():
            lower = text.lower()
            ota: bool | None = None
            if "ota" in lower or "over-the-air" in lower:
                ota = "not supported" not in lower
            if model == "ECU-850b" and "includes all features" in lower:
                ota = True
            records[model] = ModelRecord(
                model=model,
                series="ECU-700" if model == "ECU-750" else "ECU-800",
                source=SOURCE_FILES[model],
                specs=self._parse_table(text),
                ota_supported=ota,
                text=text,
            )
        return records

    def records_for(self, models: list[str]) -> list[ModelRecord]:
        """Return records in requested model order."""

        return [self.records[model] for model in models if model in self.records]

