"""Verbalized confidence and per-digit logprob weighting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import experiment_cli as cli
import pyarrow.parquet as pq
import pytest
from conftest import REPO_ROOT


def _position(sampled: str, probabilities: dict[str, float]) -> dict:
    return {
        "token": sampled,
        "logprob": math.log(probabilities[sampled]),
        "top_logprobs": [(token, math.log(probability)) for token, probability in probabilities.items()],
    }


def test_verbalized_confidence_uses_each_digit_distribution() -> None:
    parsed = {"label": "care", "confidence_tens": 9, "confidence_units": 5}
    positions = [
        {"token": "{", "logprob": 0.0, "top_logprobs": [("{", 0.0)]},
        _position("9", {"8": 0.2, "9": 0.8}),
        _position("5", {"4": 0.25, "5": 0.75}),
        {"token": "}", "logprob": 0.0, "top_logprobs": [("}", 0.0)]},
    ]

    enriched, errors = cli.verbalized_confidence(parsed, positions)

    assert errors == []
    assert enriched["verbalized_confidence"] == pytest.approx(0.95)
    assert enriched["logprob_weighted_confidence"] == pytest.approx(0.9275)
    assert enriched["confidence_digit_probability_mass"] == pytest.approx({"tens": 1.0, "units": 1.0})
    assert enriched["confidence_digit_logprobs"]["tens"]["9"] == pytest.approx(math.log(0.8))


def test_digit_logprobs_combine_tokenizer_spellings() -> None:
    position = {
        "token": " 9",
        "logprob": math.log(0.5),
        "top_logprobs": [
            ("9", math.log(0.2)),
            (" 9", math.log(0.5)),
            ('":9,', math.log(0.1)),
            ("8", math.log(0.2)),
        ],
    }

    scores = cli.digit_logprobs(position)

    assert math.exp(scores["9"]) == pytest.approx(0.8)
    assert math.exp(scores["8"]) == pytest.approx(0.2)


def test_missing_digit_position_is_an_explicit_validation_failure() -> None:
    parsed = {"label": "care", "confidence_tens": 9, "confidence_units": 5}
    enriched, errors = cli.verbalized_confidence(parsed, [_position("9", {"9": 1.0})])

    assert enriched == parsed
    assert errors and errors[0].startswith("confidence_logprobs_missing:")


def test_openai_position_extraction_keeps_sampled_token() -> None:
    content = [
        {
            "token": "9",
            "logprob": -0.1,
            "top_logprobs": [{"token": "8", "logprob": -2.0}],
        }
    ]

    positions = cli.extract_position_logprobs(content)

    assert positions[0]["token"] == "9"
    assert ("9", -0.1) in positions[0]["top_logprobs"]
    assert ("8", -2.0) in positions[0]["top_logprobs"]


def test_request_includes_schema_and_generation_logprobs(tmp_path: Path) -> None:
    config = cli.load_config(
        REPO_ROOT / "experiments" / "matrix_smoke.yaml",
        [f"output.directory={tmp_path}"],
        check_files=True,
    )
    lane = cli.select_dataset(config, "nested_json")
    variant = {
        "id": "verbalized_confidence",
        "result_type": "single_label_verbalized_confidence",
        "request_mode": "generate_with_logprobs",
        "max_tokens": 64,
        "top_logprobs": 20,
        "prompts": [
            "experiment-cli/prompt/system.md",
            "experiment-cli/prompt/context.md",
            "experiment-cli/prompt/task-verbalized-confidence.md",
            "experiment-cli/prompt/output-json.md",
            "experiment-cli/prompt/input.md",
        ],
        "validation": {
            "schema": "experiment-cli/prompt/schema-verbalized-confidence.json",
            "enum_from": "dataset_labels",
        },
    }
    schema = cli.variant_schema(lane, variant)
    row = cli.rows_for_source(lane, lane["input"], 1)[0]

    body = cli.request_for_row(lane, variant, row, schema)["body"]

    assert body["logprobs"] is True
    assert body["top_logprobs"] == 20
    assert body["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"] == ["care", "harm"]


def test_fake_backend_writes_both_confidences_to_parquet(tmp_path: Path) -> None:
    config = cli.load_config(
        REPO_ROOT / "experiments" / "matrix_smoke.yaml",
        [f"output.directory={tmp_path / 'matrix'}"],
        check_files=True,
    )
    lane = cli.select_dataset(config, "nested_json")
    lane["variants"] = [
        {
            "id": "verbalized_confidence",
            "result_type": "single_label_verbalized_confidence",
            "request_mode": "generate_with_logprobs",
            "max_tokens": 64,
            "top_logprobs": 20,
            "fake_response": {"label": "care", "confidence_tens": 7, "confidence_units": 5},
            "prompts": [
                "experiment-cli/prompt/system.md",
                "experiment-cli/prompt/context.md",
                "experiment-cli/prompt/task-verbalized-confidence.md",
                "experiment-cli/prompt/output-json.md",
                "experiment-cli/prompt/input.md",
            ],
            "validation": {
                "schema": "experiment-cli/prompt/schema-verbalized-confidence.json",
                "enum_from": "dataset_labels",
            },
        }
    ]

    manifest = cli.run(lane, cli.FakeBackend(), row_limit=2)
    rows = pq.read_table(Path(lane["output"]["directory"]) / "results.parquet").to_pylist()
    parsed = json.loads(rows[0]["parsed_output"])

    assert manifest["result_rows"] == 2
    assert rows[0]["result_type"] == "single_label_verbalized_confidence"
    assert parsed["verbalized_confidence"] == pytest.approx(0.75)
    assert parsed["logprob_weighted_confidence"] == pytest.approx(0.75)
    assert parsed["confidence_digit_probability_mass"] == {"tens": 1.0, "units": 1.0}
