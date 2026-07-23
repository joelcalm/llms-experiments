"""The API backend must constrain generation server-side, like every other path.

vLLM uses StructuredOutputsParams and `prepare` sends `response_format` in each
run-batch request. The live API backend used to send neither, so it was the only
path asking for schema-valid JSON with prompt text alone.
"""

from __future__ import annotations

import json
from typing import Any

import experiment_cli as cli
import pytest
from conftest import REPO_ROOT

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["care", "purity"]}},
    "required": ["label"],
    "additionalProperties": False,
}


class RecordingSession:
    """Stands in for `requests`, capturing the payload and replying 200."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, url: str, headers: dict, json: dict, timeout: float):
        del url, headers, timeout
        self.payloads.append(json)
        return self

    status_code = 200
    text = ""

    def json(self) -> dict:
        import json as json_module

        return {
            "choices": [{"message": {"content": json_module.dumps({"label": "care"})}}],
            "usage": {"completion_tokens": 3},
        }


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch):
    def _make(**model: Any) -> tuple[cli.OpenAICompatibleBackend, RecordingSession]:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        instance = cli.OpenAICompatibleBackend({"name": "m", "backend": "openai_compatible", **model})
        session = RecordingSession()
        instance.requests = session
        return instance, session

    return _make


def test_generate_mode_sends_the_schema_as_response_format(backend) -> None:
    api, session = backend()
    variant = {"id": "single_label_json", "request_mode": "generate", "max_tokens": 64, "_schema": SCHEMA}

    responses = api.generate(["prompt"], variant)

    payload = session.payloads[0]
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "single_label_json", "schema": SCHEMA, "strict": True},
    }
    assert json.loads(responses[0].raw) == {"label": "care"}
    assert responses[0].backend_error is None


def test_the_schema_sent_is_the_enum_substituted_one(backend, tmp_path) -> None:
    """The engine hands the backend `_schema`, which is variant_schema()'s output.

    The model constraint must match the schema used to validate its response.
    """
    config = cli.load_config(
        REPO_ROOT / "experiments" / "matrix_smoke.yaml",
        [f"output.directory={tmp_path}"],
        check_files=True,
    )
    lane = cli.select_dataset(config, cli.dataset_entries(config)[0][0])
    variant = cli.materialize_variant(lane, next(v for v in lane["variants"] if v["id"] == "single_label_json"))

    schema = cli.variant_schema(lane, variant)
    api, session = backend()
    api.generate(["prompt"], {**variant, "_schema": schema})

    sent = session.payloads[0]["response_format"]["json_schema"]["schema"]
    assert sent["properties"]["label"]["enum"] == lane["run"]["dataset_labels"]
    assert "alpha" not in sent["properties"]["label"]["enum"], "the placeholder enum must not reach the model"


def test_candidate_logprobs_mode_does_not_send_response_format(backend) -> None:
    """Scoring variants read top_logprobs from a single token; JSON would break them."""
    api, session = backend()
    variant = {"id": "logits", "request_mode": "candidate_logprobs", "candidates": ["A", "B"], "_schema": SCHEMA}

    api.generate(["prompt"], variant)

    payload = session.payloads[0]
    assert "response_format" not in payload
    assert payload["logprobs"] is True
    assert payload["max_tokens"] == 1


def test_generate_with_logprobs_keeps_schema_and_digit_positions(backend) -> None:
    api, session = backend()
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["care"]},
            "confidence_tens": {"type": "integer", "minimum": 0, "maximum": 9},
            "confidence_units": {"type": "integer", "minimum": 0, "maximum": 9},
        },
        "required": ["label", "confidence_tens", "confidence_units"],
        "additionalProperties": False,
    }
    variant = {
        "id": "verbalized_confidence",
        "request_mode": "generate_with_logprobs",
        "top_logprobs": 20,
        "_schema": schema,
    }
    payload = {"label": "care", "confidence_tens": 9, "confidence_units": 5}
    session.json = lambda: {
        "choices": [
            {
                "message": {"content": json.dumps(payload)},
                "logprobs": {
                    "content": [
                        {
                            "token": "9",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "9", "logprob": -0.1},
                                {"token": "8", "logprob": -2.4},
                            ],
                        },
                        {
                            "token": "5",
                            "logprob": -0.2,
                            "top_logprobs": [
                                {"token": "5", "logprob": -0.2},
                                {"token": "4", "logprob": -1.9},
                            ],
                        },
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 12},
    }

    response = api.generate(["prompt"], variant)[0]
    parsed, errors = cli.interpret_response(response, schema, variant["request_mode"])

    sent = session.payloads[0]
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 20
    assert sent["response_format"]["json_schema"]["schema"] == schema
    assert errors == []
    assert parsed["verbalized_confidence"] == pytest.approx(0.95)
    assert parsed["logprob_weighted_confidence"] < parsed["verbalized_confidence"]


def test_structured_outputs_can_be_disabled_for_endpoints_that_reject_it(backend) -> None:
    api, session = backend(api_structured_outputs=False)
    variant = {"id": "v", "request_mode": "generate", "_schema": SCHEMA}

    api.generate(["prompt"], variant)

    assert "response_format" not in session.payloads[0]
