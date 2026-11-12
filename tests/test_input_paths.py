"""Regression tests for environment-backed input paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from llms_experiments.configuration import load_config
from llms_experiments.inputs import create_input_reader


def test_input_reader_expands_environment_path(monkeypatch, tmp_path: Path) -> None:
    """Readers must use the same environment expansion as config validation."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    input_path = data_root / "rows.csv"
    input_path.write_text("id,text\n1,hello\n", encoding="utf-8")
    monkeypatch.setenv("DATA_ROOT", str(data_root))

    reader = create_input_reader(
        {
            "path": "${DATA_ROOT}/rows.csv",
            "format": "csv",
            "id_column": "id",
            "text_column": "text",
        },
        tmp_path / "config",
    )

    assert reader.path == input_path
    assert list(reader.iter_rows()) == [{"id": "1", "text": "hello", "_source_position": 0}]


def test_config_validation_rejects_unresolved_environment_path(tmp_path: Path) -> None:
    """A scheduler typo must fail clearly before a worker downloads a model."""
    config = tmp_path / "run.yaml"
    config.write_text(
        """
run: {id: unresolved-path}
input: {path: '${MISSING_DATA_ROOT}/rows.csv', format: csv, id_column: id, text_column: text}
model: {name: fake, backend: fake}
variants: [{id: raw, prompts: [prompt.md]}]
output: {directory: output}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unresolved environment variable.*MISSING_DATA_ROOT"):
        load_config(config, check_files=True)
