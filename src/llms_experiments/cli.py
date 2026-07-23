"""Supported llms-experiments command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import __version__
from ._core import (
    apply_resource_guard,
    batch_command,
    dataset_config,
    gpu,
    parse_batch,
    prepare,
    run,
    run_matrix,
    select_dataset,
    selected_entries,
)
from .config import load_config


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("config", type=Path)
    command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    command.add_argument("--run-id")
    command.add_argument("--model")
    command.add_argument("--backend", choices=["vllm", "openai-compatible", "fake"])
    command.add_argument("--output")
    command.add_argument("--dataset")
    command.add_argument("--datasets", help="Comma-separated dataset IDs")
    command.add_argument("--variants", help="Comma-separated variant IDs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llms-experiments", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "prepare"):
        command = commands.add_parser(name)
        _common(command)
        if name == "run":
            command.add_argument("--rows", type=int)
    parse = commands.add_parser("parse")
    _common(parse)
    parse.add_argument("--responses", type=Path, required=True)
    commands.add_parser("doctor")
    return parser


def _overrides(args: argparse.Namespace) -> list[str]:
    values = list(args.overrides)
    mappings = {
        "run_id": "run.id",
        "model": "model.name",
        "output": "output.directory",
        "backend": "model.backend",
    }
    backend_names = {"vllm": "local_vllm", "openai-compatible": "openai_compatible", "fake": "fake"}
    for attribute, key in mappings.items():
        value = getattr(args, attribute, None)
        if value is not None:
            if attribute == "backend":
                value = backend_names[value]
            values.append(f"{key}={json.dumps(value)}")
    return values


def _selected(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset:
        config = select_dataset(config, args.dataset)
    if args.variants:
        wanted = {item.strip() for item in args.variants.split(",") if item.strip()}
        known = {variant["id"] for variant in config["variants"]}
        missing = sorted(wanted - known)
        if missing:
            raise ValueError(f"Unknown variant id(s): {', '.join(missing)}")
        config["variants"] = [variant for variant in config["variants"] if variant["id"] in wanted]
    return config


def doctor() -> dict[str, Any]:
    """Return installation, backend, driver, and CUDA diagnostics."""

    snapshot = gpu()
    return {
        "version": __version__,
        "package": "ok",
        "driver_available": bool(snapshot.get("available")),
        "cuda_available": bool(snapshot.get("cuda_available", snapshot.get("available"))),
        "gpu": snapshot,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["package"] == "ok" else 2

    config = load_config(args.config, _overrides(args), check_files=True)
    config = _selected(config, args)
    apply_resource_guard(config)
    if args.command == "validate":
        print(f"Valid configuration: {config['run']['id']}")
    elif args.command == "run":
        result = (
            run_matrix(config, row_limit=args.rows, selected=_csv(args.datasets))
            if "datasets" in config
            else run(config, row_limit=args.rows)
        )
        print(json.dumps(result, indent=2))
    elif args.command == "prepare":
        if "datasets" in config:
            base_output = Path(config["output"]["directory"])
            lanes = [
                dataset_config(config, dataset_id, source, base_output)
                for dataset_id, source in selected_entries(config, _csv(args.datasets))
            ]
        else:
            lanes = [config]
        paths = [prepare(lane) for lane in lanes]
        payload = {
            "requests": [str(path) for path in paths],
            "launch_commands": [batch_command(lane) for lane in lanes],
        }
        print(json.dumps(payload, indent=2))
    else:
        if "datasets" in config and not args.dataset:
            raise ValueError("parse requires --dataset for a matrix configuration")
        print(json.dumps(parse_batch(config, args.responses), indent=2))
    return 0


def _csv(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


if __name__ == "__main__":
    raise SystemExit(main())
