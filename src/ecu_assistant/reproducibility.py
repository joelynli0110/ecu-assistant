"""Reproducibility metadata for model logging and evaluation."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

from ecu_assistant import __version__
from ecu_assistant.config import AgentConfig
from ecu_assistant.evaluation.golden_set import default_golden_set_path


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def config_snapshot(config: AgentConfig) -> dict[str, Any]:
    """Return a serializable runtime configuration snapshot."""

    return {key: _stringify(value) for key, value in asdict(config).items()}


def _run_git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_metadata(cwd: Path | None = None) -> dict[str, str | bool]:
    """Return git commit metadata when the workspace is available."""

    workspace = cwd or Path(__file__).resolve().parents[2]
    try:
        sha = _run_git(["rev-parse", "HEAD"], workspace)
        short_sha = _run_git(["rev-parse", "--short", "HEAD"], workspace)
        status = _run_git(["status", "--porcelain"], workspace)
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], workspace)
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_sha": "unknown",
            "git_short_sha": "unknown",
            "git_branch": "unknown",
            "git_dirty": "unknown",
        }
    return {
        "git_sha": sha,
        "git_short_sha": short_sha,
        "git_branch": branch,
        "git_dirty": bool(status),
    }


def _iter_files(root: Traversable) -> list[Traversable]:
    files: list[Traversable] = []
    for item in root.iterdir():
        if item.is_dir():
            files.extend(_iter_files(item))
        elif item.is_file():
            files.append(item)
    return sorted(files, key=lambda item: item.name)


def _hash_traversables(files: list[Traversable], root_label: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    manifest: list[dict[str, Any]] = []
    for item in files:
        content = item.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        manifest.append(
            {
                "path": f"{root_label}/{item.name}",
                "sha256": file_hash,
                "bytes": len(content),
            }
        )
    return {
        "sha256": digest.hexdigest(),
        "version": digest.hexdigest()[:12],
        "files": manifest,
    }


def document_metadata(docs_dir: Path | None = None) -> dict[str, Any]:
    """Hash the exact ECU source documents used by the assistant."""

    if docs_dir:
        root = docs_dir
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        ]
        digest = hashlib.sha256()
        manifest: list[dict[str, Any]] = []
        for path in files:
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content)
            manifest.append(
                {
                    "path": relative,
                    "sha256": file_hash,
                    "bytes": len(content),
                }
            )
        return {
            "source": str(root),
            "sha256": digest.hexdigest(),
            "version": digest.hexdigest()[:12],
            "files": manifest,
        }

    root = resources.files("ecu_assistant.data").joinpath("documents")
    return {
        "source": "ecu_assistant.data/documents",
        **_hash_traversables(_iter_files(root), "documents"),
    }


def evaluation_set_metadata(csv_path: Path | None = None) -> dict[str, Any]:
    """Hash the golden evaluation set."""

    path = csv_path or default_golden_set_path()
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "version": digest[:12],
        "bytes": len(content),
    }


def build_reproducibility_metadata(
    config: AgentConfig,
    eval_csv_path: Path | None = None,
) -> dict[str, Any]:
    """Build one serializable metadata bundle for MLflow and local outputs."""

    docs_dir = config.docs_dir
    documents = document_metadata(docs_dir)
    evaluation_set = evaluation_set_metadata(eval_csv_path)
    return {
        "package_version": __version__,
        "git": git_metadata(),
        "documents": documents,
        "evaluation_set": evaluation_set,
        "config": config_snapshot(config),
    }


def flatten_reproducibility_params(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return scalar MLflow params/tags for important reproducibility identifiers."""

    return {
        "package_version": metadata["package_version"],
        "git_sha": metadata["git"]["git_sha"],
        "git_short_sha": metadata["git"]["git_short_sha"],
        "git_branch": metadata["git"]["git_branch"],
        "git_dirty": str(metadata["git"]["git_dirty"]),
        "document_hash": metadata["documents"]["sha256"],
        "document_version": metadata["documents"]["version"],
        "evaluation_set_hash": metadata["evaluation_set"]["sha256"],
        "evaluation_set_version": metadata["evaluation_set"]["version"],
        "config_hash": hashlib.sha256(
            repr(sorted(metadata["config"].items())).encode("utf-8")
        ).hexdigest(),
    }
