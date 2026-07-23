"""Installable package, typed configuration, and v0.2 CLI contracts."""

from __future__ import annotations

import importlib.util
import json

import experiment_cli as shim
import pytest
from conftest import REPO_ROOT
from pydantic import ValidationError

from llms_experiments import __version__
from llms_experiments.cli import build_parser, main
from llms_experiments.config import ExperimentConfig, configuration_schema, load_config


def test_package_modules_and_version_are_importable() -> None:
    assert __version__ == "0.2.0"
    for module in ("backend", "cli", "config", "external_batch", "input", "persistence", "prompt", "runner"):
        assert importlib.util.find_spec(f"llms_experiments.{module}") is not None


def test_typed_configuration_accepts_single_and_matrix_configs() -> None:
    single = load_config(REPO_ROOT / "experiments" / "local_all_modes_smoke.yaml", check_files=True)
    matrix = load_config(REPO_ROOT / "experiments" / "matrix_smoke.yaml", check_files=True)

    assert ExperimentConfig.model_validate(single).input is not None
    assert ExperimentConfig.model_validate(matrix).datasets is not None


def test_typed_configuration_rejects_both_input_shapes() -> None:
    payload = {
        "run": {"id": "x"},
        "input": {"path": "x", "format": "csv", "id_column": "id", "text_column": "text"},
        "datasets": [
            {
                "id": "d",
                "input": {"path": "x", "format": "csv", "id_column": "id", "text_column": "text"},
            }
        ],
        "model": {"name": "fake", "backend": "fake"},
        "variants": [{"id": "v", "request_mode": "generate", "prompts": ["p.md"]}],
        "output": {"directory": "out"},
    }
    with pytest.raises(ValidationError, match="exactly one"):
        ExperimentConfig.model_validate(payload)


def test_configuration_schema_is_machine_readable() -> None:
    schema = configuration_schema()
    assert schema["title"] == "ExperimentConfig"
    assert {"run", "model", "variants", "output"}.issubset(schema["required"])
    committed = json.loads((REPO_ROOT / "docs" / "config.schema.json").read_text(encoding="utf-8"))
    assert committed["$schema"].endswith("2020-12/schema")
    assert committed["title"] == schema["title"]


def test_supported_cli_surface_has_only_five_commands() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    assert set(subparsers.choices) == {"validate", "run", "prepare", "parse", "doctor"}


def test_validate_and_doctor_run_without_a_gpu(capsys) -> None:
    assert main(["validate", str(REPO_ROOT / "experiments" / "matrix_smoke.yaml")]) == 0
    assert "Valid configuration" in capsys.readouterr().out
    assert main(["doctor"]) == 0
    assert '"package": "ok"' in capsys.readouterr().out


def test_prepare_matrix_reports_each_request_and_launch_command(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "prepare",
                str(REPO_ROOT / "experiments" / "matrix_smoke.yaml"),
                "--datasets",
                "nested_json,jsonl",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["requests"]) == len(payload["launch_commands"]) == 2
    assert all("vllm run-batch" in command for command in payload["launch_commands"])


def test_v02_shim_maps_legacy_matrix_commands() -> None:
    assert shim._forward(["run-matrix", "--config", "config.yaml", "--datasets", "x"]) == [
        "run",
        "config.yaml",
        "--datasets",
        "x",
    ]
