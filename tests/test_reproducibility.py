"""Reproducibility metadata tests."""

from pathlib import Path

from ecu_assistant.agent.graph import ECUEngineeringAgent
from ecu_assistant.config import AgentConfig
from ecu_assistant.evaluation.run_eval import evaluate_golden_set
from ecu_assistant.reproducibility import (
    build_reproducibility_metadata,
    config_snapshot,
    document_metadata,
    evaluation_set_metadata,
    flatten_reproducibility_params,
)


def test_config_snapshot_is_serializable():
    config = AgentConfig(
        docs_dir=Path("docs"),
        retrieval_k=7,
        high_confidence_threshold=0.91,
    )

    snapshot = config_snapshot(config)

    assert snapshot["docs_dir"] == "docs"
    assert snapshot["retrieval_k"] == 7
    assert snapshot["high_confidence_threshold"] == 0.91


def test_document_hash_changes_with_content(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("alpha", encoding="utf-8")

    first = document_metadata(docs_dir)
    second = document_metadata(docs_dir)
    (docs_dir / "a.md").write_text("alpha changed", encoding="utf-8")
    third = document_metadata(docs_dir)

    assert first["sha256"] == second["sha256"]
    assert first["version"] == first["sha256"][:12]
    assert first["sha256"] != third["sha256"]
    assert first["files"][0]["path"] == "a.md"


def test_evaluation_set_metadata_uses_content_hash(tmp_path):
    eval_path = tmp_path / "golden.csv"
    eval_path.write_text("Question_ID,Question\n1,hello\n", encoding="utf-8")

    metadata = evaluation_set_metadata(eval_path)

    assert metadata["path"] == str(eval_path)
    assert metadata["version"] == metadata["sha256"][:12]
    assert metadata["bytes"] > 0


def test_reproducibility_metadata_contains_required_ids():
    metadata = build_reproducibility_metadata(AgentConfig())
    params = flatten_reproducibility_params(metadata)

    assert metadata["git"]["git_sha"]
    assert metadata["documents"]["sha256"]
    assert metadata["evaluation_set"]["version"]
    assert metadata["config"]["retrieval_k"] == 4
    assert params["git_sha"] == metadata["git"]["git_sha"]
    assert params["document_hash"] == metadata["documents"]["sha256"]
    assert params["evaluation_set_version"] == metadata["evaluation_set"]["version"]
    assert params["config_hash"]


def test_evaluation_result_includes_reproducibility_metadata():
    result = evaluate_golden_set(ECUEngineeringAgent(AgentConfig()))
    metadata = result["metadata"]

    assert metadata["git"]["git_sha"]
    assert metadata["documents"]["sha256"]
    assert metadata["evaluation_set"]["sha256"]
    assert metadata["config"]["high_confidence_threshold"] == 0.9
