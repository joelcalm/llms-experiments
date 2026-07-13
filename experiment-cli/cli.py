#!/usr/bin/env python3
"""Configuration-driven local LLM experiments.

The runner contains only execution mechanics.  Prompt text, output schemas,
candidate sets, and variants are entirely supplied by YAML and adjacent files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config, root_path
from dataset import load_rows
from engine import create_backend, synchronize_cuda
from logging_utils import EventLogger, gpu_snapshot
from manifest import write_manifest
from output import load_existing, serialise, write_results
from prompts import compose
from retry import retry_response
from tuning import tune_batch
from validation import load_schema, validate_response


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    run_id = config["run"]["id"]
    output = root_path(config, config["output"]["directory"])
    logging = config.get("logging", {})
    log = root_path(config, logging.get("file", f"logs/{run_id}.log"))
    events = root_path(config, logging.get("events", f"logs/{run_id}.events.jsonl"))
    return output, log, events


def _correction_template(config: dict[str, Any]) -> str | None:
    retry = config.get("validation", {}).get("retry", {})
    template = retry.get("correction_prompt")
    return root_path(config, template).read_text(encoding="utf-8") if template else None


def run(config: dict[str, Any], *, backend: Any | None = None, supplied_responses: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    output_dir, log_path, event_path = _paths(config)
    event_log = EventLogger(log_path, event_path, config.get("logging", {}).get("level", "INFO"))
    rows = load_rows(root_path(config, config["input"]["path"]), config["input"]["format"], config["input"]["id_column"], config["input"]["text_column"])
    existing = load_existing(output_dir)
    manifest_path = output_dir / "manifest.json"
    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    completed = {(str(row["variant_id"]), str(row["input_row_id"])) for row in existing}
    result_rows = list(existing)
    root = Path(config["_root"])
    correction_template = _correction_template(config)
    model = backend
    owns_backend = False
    variant_manifest: dict[str, Any] = dict(previous_manifest.get("variants", {}))
    try:
        for variant in config["variants"]:
            pending = [row for row in rows if (variant["id"], str(row[config["input"]["id_column"]])) not in completed]
            if not pending:
                event_log.emit("variant_resumed", variant=variant["id"], skipped=len(rows))
                continue
            if model is None and supplied_responses is None:
                model = create_backend(config["model"])
                owns_backend = True
            schema_path = variant.get("validation", {}).get("schema")
            schema = load_schema(root_path(config, schema_path) if schema_path else None)
            request_variant = {**variant, "_json_schema": schema}
            values_for = lambda row: {"text": row[config["input"]["text_column"]], "row_id": row[config["input"]["id_column"]],
                                      "output_schema": json.dumps(schema or {}, sort_keys=True)}
            prompts = [compose(root, variant["prompts"], values_for(row)) for row in pending]
            batch_config = dict(config.get("batch", {}))
            max_num_seqs = int(config["model"].get("max_num_seqs", max(batch_config.get("candidates", [1]))))
            batch_config["candidates"] = [size for size in batch_config.get("candidates", [1]) if int(size) <= max_num_seqs]
            if not batch_config["candidates"]:
                batch_config["candidates"] = [max_num_seqs]
            if supplied_responses is None:
                batch_size, tuning = tune_batch(model, request_variant, prompts, batch_config, event_log.emit,
                                                 synchronize=bool(config["model"].get("synchronize_cuda", False)))
            else:
                batch_size, tuning = max(batch_config["candidates"]), []
                event_log.emit("batch_parse_selected", variant=variant["id"], batch_size=batch_size)
            variant_manifest[variant["id"]] = {"selected_batch_size": batch_size, "tuning": tuning, "pending_rows": len(pending)}
            event_log.emit("variant_started", variant=variant["id"], rows=len(pending), batch_size=batch_size)
            for start in range(0, len(pending), batch_size):
                chunk_rows, chunk_prompts = pending[start:start + batch_size], prompts[start:start + batch_size]
                synchronize_cuda(bool(config["model"].get("synchronize_cuda", False)))
                begin = time.perf_counter()
                if supplied_responses is None:
                    responses = model.generate(chunk_prompts, request_variant)
                else:
                    from engine import Response
                    missing = [(variant["id"], str(row[config["input"]["id_column"]])) for row in chunk_rows
                               if (variant["id"], str(row[config["input"]["id_column"]])) not in supplied_responses]
                    if missing:
                        raise ValueError(f"Missing parsed response(s), first: {missing[0]}")
                    responses = [Response(supplied_responses[(variant["id"], str(row[config["input"]["id_column"]]))], 0) for row in chunk_rows]
                synchronize_cuda(bool(config["model"].get("synchronize_cuda", False)))
                elapsed = max(time.perf_counter() - begin, 1e-9)
                for row, prompt, response in zip(chunk_rows, chunk_prompts, responses):
                    parsed, errors = validate_response(response.raw, schema)
                    if variant["request_mode"] == "candidate_logprobs":
                        parsed = {"candidates": response.candidate_logprobs or {}}
                    attempts = 1
                    if errors:
                        event_log.emit("validation_failed", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]]), errors=errors)
                        def request(correction: str) -> str:
                            if model is None:
                                raise RuntimeError("parse received an invalid response requiring retry; run prepare for correction requests or supply a backend")
                            return model.generate([correction], request_variant)[0].raw
                        if model is None:
                            raw = response.raw
                            event_log.emit("retry_deferred_for_parse", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]]))
                        else:
                            event_log.emit("retry_started", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]]))
                            raw, parsed, errors, attempts = retry_response(response.raw, errors, config.get("validation", {}).get("retry", {}), correction_template,
                                                                                request, values_for(row), lambda raw: validate_response(raw, schema))
                            event_log.emit("retry_completed", variant=variant["id"], input_row_id=str(row[config["input"]["id_column"]]),
                                           attempts=attempts, validation_status="valid" if not errors else "invalid")
                    else:
                        raw = response.raw
                    snapshot = gpu_snapshot()
                    result_rows.append({
                        "run_id": config["run"]["id"], "variant_id": variant["id"], "input_row_id": str(row[config["input"]["id_column"]]),
                        "source_position": row["_source_position"], "input_text": str(row[config["input"]["text_column"]]) if config["output"].get("include_text") else None,
                        "prompt_hash": _hash(prompt), "config_hash": _hash(json.dumps(variant, sort_keys=True)),
                        "attempt_count": attempts, "raw_response": raw if config["output"].get("include_raw_response", True) else None,
                        "parsed_output": serialise(parsed), "validation_status": "valid" if not errors else "invalid", "validation_errors": serialise(errors),
                        "final_status": "completed" if not errors else "failed_validation", "batch_size": batch_size, "latency_seconds": elapsed / len(chunk_rows),
                        "rows_per_second": len(chunk_rows) / elapsed, "token_count": response.token_count,
                        "gpu_snapshot": serialise(snapshot), "candidate_logprobs": serialise(response.candidate_logprobs),
                    })
                event_log.emit("batch_completed", variant=variant["id"], rows=len(chunk_rows), latency_seconds=elapsed,
                               rows_per_second=len(chunk_rows) / elapsed, gpu=gpu_snapshot())
            write_results(output_dir, result_rows)
            event_log.emit("variant_completed", variant=variant["id"], rows=len(pending))
    finally:
        if owns_backend and model is not None:
            model.close()
    manifest = {"run_id": config["run"]["id"], "input_rows": len(rows), "result_rows": len(result_rows), "model": config["model"],
                "variants": variant_manifest, "gpu_preflight": gpu_snapshot(), "resume_skipped_rows": len(existing), "event_log": str(event_path)}
    write_manifest(output_dir, config, manifest)
    event_log.emit("run_completed", result_rows=len(result_rows), gpu=gpu_snapshot())
    return manifest


def prepare(config: dict[str, Any]) -> Path:
    output_dir, _, _ = _paths(config)
    rows = load_rows(root_path(config, config["input"]["path"]), config["input"]["format"], config["input"]["id_column"], config["input"]["text_column"])
    path = output_dir / "raw" / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(config["_root"])
    with path.open("w", encoding="utf-8") as handle:
        for variant in config["variants"]:
            schema_path = variant.get("validation", {}).get("schema")
            schema = load_schema(root_path(config, schema_path) if schema_path else None)
            for row in rows:
                values = {"text": row[config["input"]["text_column"]], "row_id": row[config["input"]["id_column"]], "output_schema": json.dumps(schema or {})}
                handle.write(json.dumps({"custom_id": f"{variant['id']}:{row[config['input']['id_column']]}", "variant_id": variant["id"],
                                         "input_row_id": str(row[config["input"]["id_column"]]), "prompt": compose(root, variant["prompts"], values)}) + "\n")
    return path


def parse(config: dict[str, Any], responses: Path) -> dict[str, Any]:
    source = responses / "responses.jsonl" if responses.is_dir() else responses
    values: dict[tuple[str, str], str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        variant, row_id = item.get("variant_id"), str(item.get("input_row_id"))
        if not variant and "custom_id" in item:
            variant, row_id = item["custom_id"].split(":", 1)
        values[(variant, row_id)] = str(item.get("response", item.get("raw_response", "")))
    return run(config, supplied_responses=values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    parse_command = commands.add_parser("parse")
    parse_command.add_argument("--config", required=True)
    parse_command.add_argument("--responses", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        print(f"Valid configuration: {config['run']['id']}")
    elif args.command == "prepare":
        print(prepare(config))
    elif args.command == "parse":
        print(json.dumps(parse(config, Path(args.responses)), indent=2))
    else:
        print(json.dumps(run(config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
