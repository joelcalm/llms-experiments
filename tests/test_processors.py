"""Processor pipeline compilation, evidence handling, and failure semantics."""

from __future__ import annotations

import json
import math

import pytest

from llms_experiments.processors import (
    ConfigurationDefaultWarning,
    RawModelResponse,
    TokenCandidate,
    TokenPosition,
    create_processor,
)


def test_omitted_processor_warns_and_returns_raw_output(tmp_path) -> None:
    with pytest.warns(ConfigurationDefaultWarning, match="identity"):
        processor = create_processor({"id": "raw"}, root=tmp_path)

    result = processor.process(RawModelResponse("unparsed output", 3), {"_source_position": 0})

    assert processor.result_type == "raw"
    assert result.value == "unparsed output"
    assert result.valid


def test_structured_pipeline_decodes_and_validates(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"label": {"type": "string", "enum": ["care"]}},
                "required": ["label"],
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    processor = create_processor(
        {
            "id": "single",
            "processor": {
                "result": "single_label",
                "stages": [
                    {"type": "json_decode"},
                    {"type": "json_schema", "schema": str(schema)},
                ],
            },
        },
        root=tmp_path,
    )

    valid = processor.process(RawModelResponse('{"label":"care"}', 2), {"_source_position": 0})
    invalid = processor.process(RawModelResponse('{"label":"harm"}', 2), {"_source_position": 1})

    assert valid.value == {"label": "care"}
    assert valid.valid
    assert invalid.errors[0].code == "schema_validation_error"
    assert invalid.value == {"label": "harm"}


def test_candidate_pipeline_aggregates_tokenizer_spellings(tmp_path) -> None:
    processor = create_processor(
        {
            "id": "codes",
            "processor": {
                "result": "categorical_logprobs",
                "stages": [{"type": "candidate_logprobs", "candidates": ["A", "B"]}],
            },
        },
        root=tmp_path,
    )
    position = TokenPosition(
        token=" A",
        logprob=math.log(0.5),
        alternatives=(
            TokenCandidate("A", math.log(0.2)),
            TokenCandidate(" A", math.log(0.5)),
            TokenCandidate("B", math.log(0.3)),
        ),
    )

    result = processor.process(RawModelResponse("A", 1, (position,)), {"_source_position": 0})

    assert math.exp(result.candidate_scores["A"]) == pytest.approx(0.7)
    assert math.exp(result.candidate_scores["B"]) == pytest.approx(0.3)


def test_fan_out_prepares_stable_target_rows(tmp_path) -> None:
    processor = create_processor(
        {
            "id": "labels",
            "processor": {
                "result": "label_yes_no_logprobs",
                "stages": [
                    {"type": "fan_out", "over": "dataset_labels"},
                    {"type": "candidate_logprobs", "candidates": ["yes", "no"]},
                ],
            },
        },
        root=tmp_path,
        dataset_labels=["care", "harm"],
    )

    rows = list(processor.prepare_rows([{"id": "x", "_source_position": 4}]))

    assert [(row["_target_label"], row["_source_position"]) for row in rows] == [("care", 8), ("harm", 9)]


def test_stage_failure_stops_dependent_stages_and_preserves_raw(tmp_path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}', encoding="utf-8")
    processor = create_processor(
        {
            "id": "confidence",
            "processor": {
                "result": "single_label_verbalized_confidence",
                "stages": [
                    {"type": "json_decode"},
                    {"type": "json_schema", "schema": str(schema)},
                    {"type": "verbalized_confidence"},
                ],
            },
        },
        root=tmp_path,
    )
    response = RawModelResponse("not json", 2)

    result = processor.process(response, {"_source_position": 0})

    assert result.errors[0].stage == "json_decode"
    assert len(result.errors) == 1
    assert result.value is None
    assert response.text == "not json"


def test_pipeline_rejects_identity_composition(tmp_path) -> None:
    with pytest.raises(ValueError, match="identity must be the only stage"):
        create_processor(
            {
                "id": "invalid",
                "processor": {
                    "result": "raw",
                    "stages": [{"type": "identity"}, {"type": "json_decode"}],
                },
            },
            root=tmp_path,
        )
