"""Regression tests for environment-backed input paths."""

from __future__ import annotations

from pathlib import Path

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
