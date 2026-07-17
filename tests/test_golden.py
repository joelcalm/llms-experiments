"""End-to-end golden snapshots captured with the fake backend.

These lock the observable output of the runner so the refactor can be shown to
change structure without changing behaviour.  They need no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import assert_golden, read_manifest, read_results, run_cli

SMOKE = "experiments/local_all_modes_smoke.yaml"
MATRIX = "experiments/ministral_all_datasets.yaml"


def test_run_smoke(tmp_path: Path, golden_update: bool) -> None:
    out = tmp_path / "run"
    run_cli("run", "--config", SMOKE, "--backend", "fake", "--output", str(out))
    assert_golden(
        "run_smoke",
        {"rows": read_results(out), "manifest": read_manifest(out)},
        golden_update,
    )


def test_run_smoke_is_resumable(tmp_path: Path) -> None:
    """A second run must skip everything the first one completed."""
    out = tmp_path / "run"
    first = json.loads(run_cli("run", "--config", SMOKE, "--backend", "fake", "--output", str(out)))
    second = json.loads(run_cli("run", "--config", SMOKE, "--backend", "fake", "--output", str(out)))
    assert first["result_rows"] == 5 * 128
    assert second["resume_skipped_rows"] == first["result_rows"]
    assert second["result_rows"] == first["result_rows"]


def test_run_matrix(tmp_path: Path, golden_update: bool) -> None:
    """Covers all five dataset lanes: nested_json, jsonl, paired_tsv and filtered csv."""
    out = tmp_path / "matrix"
    run_cli("run-matrix", "--config", MATRIX, "--backend", "fake", "--rows", "8", "--output", str(out))
    payload = {
        "summary": read_manifest(out, "matrix_manifest.json"),
        "datasets": {
            path.name: read_results(path)
            for path in sorted(out.glob("dataset=*"))
            if (path / "results.parquet").exists()
        },
    }
    assert_golden("run_matrix", payload, golden_update)


def test_prepare_parse_roundtrip(tmp_path: Path, golden_update: bool) -> None:
    """prepare -> synthetic responses -> parse. This is the external-batch path."""
    out = tmp_path / "batch"
    run_cli("prepare", "--config", SMOKE, "--output", str(out))
    requests_path = out / "requests.jsonl"
    responses_path = out / "responses.jsonl"
    responses_path.write_text(
        "".join(
            json.dumps(fake_response(json.loads(line))) + "\n"
            for line in requests_path.read_text().splitlines()
            if line.strip()
        ),
        encoding="utf-8",
    )
    run_cli("parse", "--config", SMOKE, "--responses", str(responses_path), "--output", str(out))
    assert_golden(
        "parse_smoke",
        {"rows": read_results(out), "manifest": read_manifest(out)},
        golden_update,
    )


def test_prepare_parse_roundtrip_honours_enum_from(tmp_path: Path, golden_update: bool) -> None:
    """The batch path must validate against the schema it actually sent.

    `prepare` substitutes `enum_from: dataset_labels` into the schema, so the
    model is constrained to the real labels. `parse` used to re-load the raw
    schema off disk and judge those labels against its placeholder enum
    (alpha/beta/gamma), marking every row invalid and emitting a retry for it.

    The smoke config cannot catch this: its dataset_labels *are* alpha/beta/gamma
    and it sets no enum_from, so the two schemas coincide. This lane is a config
    where they genuinely differ.
    """
    out = tmp_path / "batch"
    dataset = "mftc"
    # The lane's own input, not the top-level one: dataset_config() replaces
    # config["input"] wholesale, so `--set input.limit` never reaches a lane.
    limit = ("--set", "datasets.0.input.limit=8")
    run_cli("prepare-matrix", "--config", MATRIX, "--datasets", dataset, *limit, "--output", str(out))
    lane = out / f"dataset={dataset}"
    requests = [json.loads(line) for line in (lane / "requests.jsonl").read_text().splitlines() if line.strip()]

    enums = {
        json.dumps(request["body"]["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"])
        for request in requests
        if request["custom_id"].startswith("single_label_json:")
    }
    assert enums, "expected the enum_from variant to carry a response_format schema"
    assert "alpha" not in enums.pop(), "prepare must send the substituted enum, not the placeholder"

    responses_path = lane / "responses.jsonl"
    responses_path.write_text(
        "".join(json.dumps(schema_obeying_response(request)) + "\n" for request in requests),
        encoding="utf-8",
    )
    run_cli(
        "parse",
        "--config",
        MATRIX,
        "--dataset",
        dataset,
        *limit,
        "--responses",
        str(responses_path),
        "--output",
        str(out),
    )

    rows = [row for row in read_results(lane) if row["variant_id"] in {"single_label_json", "multi_label_json"}]
    assert rows, "expected rows for the enum_from variants"
    assert {row["validation_status"] for row in rows} == {"valid"}, (
        "a response the model was constrained to produce must not be judged invalid"
    )
    assert read_manifest(lane)["retry_requests"] == 0, "valid rows must not queue retries"
    assert_golden("parse_enum_from", {"rows": rows, "manifest": read_manifest(lane)}, golden_update)


def schema_obeying_response(request: dict) -> dict:
    """Answer using the schema the request carried, as a real batch server does.

    Deriving the answer from `response_format` rather than hard-coding it is the
    whole point: it reproduces structured outputs, where the model can only emit
    values the sent schema permits.
    """
    body = request["body"]
    schema = (body.get("response_format") or {}).get("json_schema", {}).get("schema")
    if body.get("logprobs"):
        return fake_response(request)
    properties = (schema or {}).get("properties", {})
    payload: dict = {}
    for field, definition in properties.items():
        if definition.get("type") == "array":
            payload[field] = [definition.get("items", {}).get("enum", ["x"])[0]]
        elif "enum" in definition:
            payload[field] = definition["enum"][0]
        elif definition.get("type") == "integer":
            payload[field] = definition.get("minimum", 1)
        else:
            payload[field] = "x"
    return {
        "custom_id": request["custom_id"],
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"completion_tokens": 5},
            },
        },
    }


def fake_response(request: dict) -> dict:
    """Mimic one vLLM `run-batch` response line for a prepared request."""
    body = request["body"]
    variant = request["custom_id"].split(":", 1)[0]
    payloads = {
        "single_label_json": {"label": "alpha"},
        "multi_label_json": {"labels": ["alpha"]},
        "ordinal_score_json": {"score": 3},
    }
    message: dict = {"content": json.dumps(payloads.get(variant, {"label": "alpha"}))}
    choice: dict = {"message": message}
    if body.get("logprobs"):
        # candidate_logprobs variants read top_logprobs rather than content.
        message["content"] = "A"
        choice["logprobs"] = {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "A", "logprob": -0.1},
                        {"token": "B", "logprob": -1.2},
                        {"token": "C", "logprob": -3.4},
                        {"token": "yes", "logprob": -0.7},
                        {"token": "no", "logprob": -2.1},
                    ]
                }
            ]
        }
    return {
        "custom_id": request["custom_id"],
        "response": {"status_code": 200, "body": {"choices": [choice], "usage": {"completion_tokens": 1}}},
    }
