"""Unit coverage for the runner's pure functions and persistence helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import experiment_cli as cli
import pytest

# --- candidate logprob aggregation -------------------------------------------------


def test_aggregate_candidate_logprobs_strips_whitespace() -> None:
    scores = cli.aggregate_candidate_logprobs([(" A", -0.5), ("B", -1.5)], ["A", "B"])
    assert scores == {"A": -0.5, "B": -1.5}


def test_aggregate_candidate_logprobs_combines_duplicates_with_logsumexp() -> None:
    """' A' and 'A' are the same candidate, so their probabilities add."""
    scores = cli.aggregate_candidate_logprobs([(" A", math.log(0.3)), ("A", math.log(0.2))], ["A"])
    assert scores["A"] == pytest.approx(math.log(0.5))


def test_aggregate_candidate_logprobs_missing_candidate_is_negative_infinity() -> None:
    scores = cli.aggregate_candidate_logprobs([("A", -0.1)], ["A", "Z"])
    assert scores["Z"] == -float("inf")


# --- schema validation --------------------------------------------------------------


def test_validate_response_without_schema_passes_raw_through() -> None:
    assert cli.validate_response("anything", None) == ("anything", [])


def test_validate_response_reports_json_parse_error() -> None:
    parsed, errors = cli.validate_response("{not json", {"type": "object"})
    assert parsed is None
    assert errors and errors[0].startswith("json_parse_error:")


def test_check_schema_enforces_enum_types_and_required_keys() -> None:
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": ["a", "b"]}},
        "required": ["label"],
        "additionalProperties": False,
    }
    assert cli.validate_response(json.dumps({"label": "a"}), schema)[1] == []
    assert cli.validate_response(json.dumps({"label": "z"}), schema)[1] != []
    assert cli.validate_response(json.dumps({}), schema)[1] != []
    assert cli.validate_response(json.dumps({"label": "a", "extra": 1}), schema)[1] != []


def test_check_schema_booleans_are_not_integers() -> None:
    errors: list[str] = []
    cli.check_schema(True, {"type": "integer"}, "$", errors)
    assert errors


# --- label parsing ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ([], []),
        (["a", "b"], ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        ("a,b", ["a", "b"]),
        ("a, b ", ["a", "b"]),
        ("single", ["single"]),
    ],
)
def test_split_labels(value: object, expected: list[str]) -> None:
    assert cli._split_labels(value) == expected


# --- prompt rendering ---------------------------------------------------------------


def test_render_substitutes_known_tokens() -> None:
    assert cli.render("Text: {{text}}", {"text": "hi"}) == "Text: hi"


def test_render_rejects_unresolved_placeholder() -> None:
    with pytest.raises(ValueError, match="unresolved prompt placeholder"):
        cli.render("{{nonsense}}", {})


# --- dataset readers ----------------------------------------------------------------


def test_read_rows_csv_applies_where_filter_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("id,text,keep\n1,a,yes\n2,b,no\n3,c,yes\n", encoding="utf-8")
    rows = cli.read_rows(path, "csv", "id", "text", {"path": str(path), "where": {"keep": "yes"}})
    assert [row["id"] for row in rows] == ["1", "3"]


def test_read_rows_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "in.csv"
    path.write_text("id,other\n1,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        cli.read_rows(path, "csv", "id", "text")


def test_read_rows_jsonl_records_source_position(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    path.write_text('{"id":"a","text":"one"}\n{"id":"b","text":"two"}\n', encoding="utf-8")
    rows = cli.read_rows(path, "jsonl", "id", "text")
    assert [row["_source_position"] for row in rows] == [0, 1]


def test_read_rows_paired_tsv_joins_labels(tmp_path: Path) -> None:
    args = tmp_path / "args.tsv"
    labels = tmp_path / "labels.tsv"
    args.write_text("id\ttext\nA1\thello\n", encoding="utf-8")
    labels.write_text("id\tcare\tharm\nA1\t1\t0\n", encoding="utf-8")
    rows = cli.read_rows(args, "paired_tsv", "id", "text", {"path": str(args), "labels_path": str(labels)})
    assert rows[0]["_gold_labels"] == ["care"]


# --- config loading -----------------------------------------------------------------


def test_load_config_applies_dotted_overrides(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    loaded = cli.load_config(config, ["model.name=other", "run.id=changed"], check_files=False)
    assert loaded["model"]["name"] == "other"
    assert loaded["run"]["id"] == "changed"


def test_load_config_rejects_override_without_equals(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    with pytest.raises(ValueError, match="KEY=VALUE"):
        cli.load_config(config, ["bogus"], check_files=False)


def test_validate_config_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path, variants_duplicated=True)
    with pytest.raises(ValueError, match="unique id"):
        cli.load_config(config, check_files=False)


def test_validate_config_rejects_unknown_backend(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path, backend="wat")
    with pytest.raises(ValueError, match=r"model\.backend"):
        cli.load_config(config, check_files=False)


# --- resume index -------------------------------------------------------------------


def test_resume_index_roundtrip(tmp_path: Path) -> None:
    index = cli.ResumeIndex(tmp_path / "r.sqlite", "fp1")
    key = ("v", "row", 0)
    assert not index.contains(key)
    index.add(key)
    assert index.contains(key)
    index.close()


def test_resume_index_clears_when_fingerprint_changes(tmp_path: Path) -> None:
    """A changed config must invalidate completed work rather than resume onto it."""
    path = tmp_path / "r.sqlite"
    first = cli.ResumeIndex(path, "fp1")
    first.add(("v", "row", 0))
    first.close()

    same = cli.ResumeIndex(path, "fp1")
    assert same.contains(("v", "row", 0))
    assert not same.cleared
    same.close()

    changed = cli.ResumeIndex(path, "fp2")
    assert changed.cleared
    assert not changed.contains(("v", "row", 0))
    changed.close()


# --- parts merging ------------------------------------------------------------------


def test_merge_parts_concatenates_variant_parts(tmp_path: Path) -> None:
    writer = cli.PartWriter(tmp_path, "v1", target_rows=2)
    for position in range(4):
        writer.append(_result_row(position))
    writer.close()

    merged = cli.merge_parts(tmp_path)
    assert merged == 4
    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "v1.parquet").exists()


def test_merge_parts_with_no_parts_returns_zero(tmp_path: Path) -> None:
    assert cli.merge_parts(tmp_path) == 0


# --- per-label request expansion -----------------------------------------------------


def test_soft_multi_label_expansion_produces_unique_stable_positions() -> None:
    rows = [{"id": "a", "text": "t", "_source_position": 0}, {"id": "b", "text": "t", "_source_position": 1}]
    expanded = list(cli.expanded_rows(iter(rows), ["care", "harm"]))
    assert [
        (r["input_row_id"] if "input_row_id" in r else r["id"], r["_target_label"], r["_source_position"])
        for r in expanded
    ] == [
        ("a", "care", 0),
        ("a", "harm", 1),
        ("b", "care", 2),
        ("b", "harm", 3),
    ]


# --- helpers ------------------------------------------------------------------------


def _result_row(position: int) -> dict:
    row = dict.fromkeys(cli.RESULT_SCHEMA.names)
    row.update(
        {
            "run_id": "r",
            "dataset_id": "d",
            "variant_id": "v1",
            "input_row_id": f"row-{position}",
            "source_position": position,
            "config_hash": "hash",
            "attempt_count": 1,
            "final_status": "completed",
        }
    )
    return row


def _write_minimal_config(tmp_path: Path, *, backend: str = "fake", variants_duplicated: bool = False) -> Path:
    variant = {"id": "v1", "request_mode": "generate", "prompts": ["p.md"]}
    variants = [variant, dict(variant)] if variants_duplicated else [variant]
    payload = {
        "run": {"id": "test"},
        "input": {"path": "in.csv", "format": "csv", "id_column": "id", "text_column": "text"},
        "model": {"name": "m", "backend": backend},
        "variants": variants,
        "output": {"directory": "out"},
    }
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path
