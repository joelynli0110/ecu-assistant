"""Load, normalize, and parse ECU Markdown documents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from ecu_assistant.data.schemas import ModelRecord, SOURCE_FILES

FEATURE_ALIASES = {
    "operating temp.": "operating temperature",
    "memory (ram)": "memory",
    "can interface": "can",
    "power consumption": "power",
}

FIELD_ALIASES = {
    "processor": {
        "processor", "processors", "cpu", "cpus", "chip", "chipset",
        "clock", "clock speed", "processing unit",
    },
    "memory": {"memory", "memory capacity", "ram", "ram capacity", "sram", "lpddr4"},
    "storage": {
        "storage", "storage capacity", "flash", "internal flash", "emmc",
        "flash memory", "disk", "disk space", "disk capacity", "capacity",
    },
    "can": {
        "can", "can bus", "can interface", "can speed", "can bus speed",
        "can capability", "can capabilities", "bus throughput",
    },
    "operating voltage": {
        "operating voltage", "voltage", "voltage range", "supply voltage",
    },
    "power": {
        "power", "power consumption", "current", "current draw",
        "energy consumption", "under load", "idle consumption",
    },
    "operating temperature": {
        "operating temperature", "temperature", "temperature range",
        "max temperature", "maximum temperature", "thermal range",
    },
    "connectors": {"connector", "connectors", "ports", "physical connectors"},
    "ethernet": {"ethernet", "network interface", "networking"},
    "npu": {
        "npu", "neural processing unit", "ai accelerator", "ai acceleration",
        "ai capability", "ai capabilities", "tops",
    },
    "ota": {
        "ota", "over the air", "over-the-air", "ota update", "ota updates",
        "wireless update", "wireless updates", "firmware update",
        "firmware updates", "remote update", "remote updates",
    },
}


def normalize_text(text: str) -> str:
    """Repair known source encoding issues and normalize line endings."""

    return text.replace("\u63b3C", "°C").replace("\r\n", "\n").strip()


def _clean_markdown(value: str) -> str:
    return re.sub(r"[*_`]", "", value).strip()


def parse_spec_table(text: str) -> dict[str, str]:
    """Parse canonical specification fields from one Markdown text fragment."""

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
            specs[normalize_field(feature)] = value
    return specs


def normalize_field(field: str) -> str:
    """Resolve aliases to a canonical document field name."""

    normalized = re.sub(r"[^a-z0-9]+", " ", field.lower()).strip()
    for canonical, aliases in FIELD_ALIASES.items():
        normalized_aliases = {
            re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            for alias in aliases | {canonical}
        }
        if normalized in normalized_aliases:
            return canonical
    return FEATURE_ALIASES.get(normalized, normalized)


def detect_spec_field(
    query: str,
    records: Mapping[str, ModelRecord],
) -> str | None:
    """Detect a requested field from aliases or dynamically parsed table keys."""

    normalized_query = f" {re.sub(r'[^a-z0-9]+', ' ', query.lower()).strip()} "
    candidates: dict[str, set[str]] = {
        canonical: aliases | {canonical}
        for canonical, aliases in FIELD_ALIASES.items()
    }
    for record in records.values():
        for field in record.specs:
            canonical = normalize_field(field)
            candidates.setdefault(canonical, set()).add(field)

    matches: list[tuple[int, str]] = []
    for canonical, aliases in candidates.items():
        for alias in aliases:
            normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
            if normalized_alias and f" {normalized_alias} " in normalized_query:
                matches.append((len(normalized_alias), canonical))
    return max(matches, default=(0, ""))[1] or None


def lookup_spec(
    records: Mapping[str, ModelRecord],
    model: str,
    field: str,
) -> str | None:
    """Look up any parsed specification field for one model."""

    record = records.get(model)
    if not record:
        return None
    return record.specs.get(normalize_field(field))


def compare_specs(
    records: Mapping[str, ModelRecord],
    models: Sequence[str],
    field: str,
) -> dict[str, str | None]:
    """Look up one arbitrary field across multiple models."""

    return {model: lookup_spec(records, model, field) for model in models}


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

    def _parse_records(self) -> dict[str, ModelRecord]:
        records: dict[str, ModelRecord] = {}
        for model, text in self._texts.items():
            lower = text.lower()
            ota: bool | None = None
            if "ota" in lower or "over-the-air" in lower:
                ota = "not supported" not in lower
            if model == "ECU-850b" and "includes all features" in lower:
                ota = True
            specs = parse_spec_table(text)
            if ota is not None:
                specs["ota"] = "Supported" if ota else "Not supported"
            records[model] = ModelRecord(
                model=model,
                series="ECU-700" if model == "ECU-750" else "ECU-800",
                source=SOURCE_FILES[model],
                specs=specs,
                ota_supported=ota,
                text=text,
            )
        return records

    def records_for(self, models: list[str]) -> list[ModelRecord]:
        """Return records in requested model order."""

        return [self.records[model] for model in models if model in self.records]

    def lookup_spec(self, model: str, field: str) -> str | None:
        """Look up any parsed field for a model."""

        return lookup_spec(self.records, model, field)

    def compare_specs(
        self,
        models: Sequence[str],
        field: str,
    ) -> dict[str, str | None]:
        """Compare one parsed field across models."""

        return compare_specs(self.records, models, field)
