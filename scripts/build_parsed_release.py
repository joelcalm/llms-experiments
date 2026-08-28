#!/usr/bin/env python3
"""Build the OSF-ready parsed-result release from a source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from export_parsed_results import MFT_LABELS, SHVT_LABELS, VALUE_COLUMNS, export_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema() -> dict[str, Any]:
    """Return the public projection schema and label mapping."""

    mapping = {}
    for label in MFT_LABELS:
        mapping[f"mft__{label}"] = label
    for label in SHVT_LABELS:
        mapping[
            "shvt__"
            + "".join("_" if not char.isalnum() else char.lower() for char in label).strip("_").replace("__", "_")
        ] = label
    return {
        "schema_version": "protoethos-parsed-results-v1",
        "description": "Validated single- and multi-label JSON predictions projected to one indicator column per value.",
        "columns": {
            "model_id": "string",
            "strategy": "string",
            "lane_id": "string",
            "label_space": "string",
            "input_row_id": "string",
            "source_position": "int64",
            **{name: "uint8; values are exactly 0 or 1" for name in VALUE_COLUMNS},
        },
        "value_column_mapping": mapping,
        "mft10_labels_in_order": list(MFT_LABELS),
        "shvt20_labels_in_order": list(SHVT_LABELS),
        "compression": {"codec": "zstd", "level": 3},
        "excluded_fields": [
            "input_text",
            "raw_response",
            "parsed_output",
            "gold_labels",
            "validation_errors",
            "latency_seconds",
            "gpu_snapshot",
            "candidate_scores",
            "metrics",
            "axiom_scores",
        ],
    }


def _projection_command(source_root: str, model_id: str, strategy: str, output_file: str) -> str:
    return (
        "uv run python scripts/export_parsed_results.py "
        f"--source-root {source_root} --model-id {model_id} "
        f"--strategy {strategy} --output-file {output_file}"
    )


def _runtime_configuration(source_root: Path) -> dict[str, Any]:
    """Extract stable inference settings from the first lane manifest."""

    manifests = sorted(source_root.glob("dataset=*/manifest.json"))
    if not manifests:
        return {"warning": "source manifest unavailable"}
    lane_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    model = lane_manifest.get("model", {})
    effective = lane_manifest.get("effective_config", {})
    variants = effective.get("variants", [])
    return {
        "model_name_in_source_manifest": model.get("name"),
        "revision": model.get("revision"),
        "tokenizer_revision": model.get("tokenizer_revision"),
        "backend": model.get("backend"),
        "dtype": model.get("dtype"),
        "max_model_len": model.get("max_model_len"),
        "max_num_seqs": model.get("max_num_seqs"),
        "max_num_batched_tokens": model.get("max_num_batched_tokens"),
        "gpu_memory_utilization": model.get("gpu_memory_utilization"),
        "enforce_eager": model.get("enforce_eager"),
        "enable_chunked_prefill": model.get("enable_chunked_prefill"),
        "async_scheduling": model.get("async_scheduling"),
        "chat_template_kwargs": model.get("chat_template_kwargs"),
        "variant_max_tokens": {v.get("id"): v.get("max_tokens") for v in variants},
    }


def _readme(manifest: dict[str, Any], output_root: Path, build_command: str) -> str:
    lines = [
        "# ProtoEthos parsed results v1",
        "",
        "This release contains only schema-valid parsed predictions from the two JSON strategies. It contains no raw model responses, prompts, input text, logs, resume databases, metrics, or axiom scores.",
        "",
        "## Reproduce from a clean checkout",
        "",
        "The commands below assume that the external datasets and model access are available. Credentials must remain in the user environment and must never be placed in this repository or OSF release.",
        "",
        "```bash",
        "git clone <repository-url> llms-experiments",
        "cd llms-experiments",
        "uv sync --all-groups --extra gpu",
        "uv run llms-experiments doctor",
        "```",
        "",
        "The exact source experiment command is recorded per model below. For in-process vLLM runs, that command performs prompt rendering, inference, JSON decoding, schema validation, and Result Contract Parquet writing in one step. The processor stages are `json_decode` followed by `json_schema`; there is no separate `parse` command for this path.",
        "",
        "If `revision` is `null` in the runtime table, the original source manifest did not pin a model revision. The model identifier, configuration hashes, and observed runtime settings remain recorded, but exact weight reproducibility is not guaranteed until that revision is supplied.",
        "",
        "## Exact prompting assets",
        "",
        "The prompt is rendered in this exact order for every source row:",
        "",
        "```text",
        "experiment-cli/prompt/system.md",
        "experiment-cli/prompt/context.md",
        "experiment-cli/prompt/task-single-label.md  (single_label_json)",
        "experiment-cli/prompt/task-multi-label.md   (multi_label_json)",
        "experiment-cli/prompt/output-json.md",
        "experiment-cli/prompt/input.md",
        "```",
        "",
        "The JSON schemas are `experiment-cli/prompt/schema-single-label.json` and `experiment-cli/prompt/schema-multi-label.json`; their `dataset_labels` enumeration is materialised per lane. The exact prompt and schema hashes used by each source lane are preserved in its original `manifest.json` and summarised in the release manifest.",
        "",
        "## Runtime configuration recorded from source manifests",
        "",
        "| Model | Revision | Tokenizer revision | Dtype | Max model length | Max sequences | Max batched tokens | GPU memory fraction | Eager | Chunked prefill | Async scheduling | Variant max tokens |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for item in manifest["files"]:
        if item["strategy"] != "single_label_json":
            continue
        runtime = item.get("runtime_configuration", {})
        lines.append(
            f"| `{item['model_id']}` | `{runtime.get('revision')}` | `{runtime.get('tokenizer_revision')}` | `{runtime.get('dtype')}` | {runtime.get('max_model_len')} | {runtime.get('max_num_seqs')} | {runtime.get('max_num_batched_tokens')} | {runtime.get('gpu_memory_utilization')} | {runtime.get('enforce_eager')} | {runtime.get('enable_chunked_prefill')} | {runtime.get('async_scheduling')} | `{runtime.get('variant_max_tokens')}` |"
        )
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(
        [
            "For sources copied from Artemisa, the exact read-only acquisition command is recorded in `source_manifest.json` and `manifest.json`; the copied files are then projected locally with the commands below.",
            "",
            "For external batch runs, the equivalent parse phase is:",
            "",
            "```bash",
            "uv run llms-experiments prepare CONFIG.yaml",
            "# exact external inference command used for the batch provider",
            "uv run llms-experiments parse CONFIG.yaml --responses /path/to/responses.jsonl",
            "```",
            "",
            "## Build command used for this release",
            "",
            "```bash",
            build_command,
            "```",
            "",
            "The compact projection command for each output file is listed in the table below. It reads only the final `single_label_json.parquet` or `multi_label_json.parquet` files under the source root, keeps valid completed rows, and writes the fixed 30-value indicator schema.",
            "",
            "## Output layout",
            "",
            "```text",
            "single_label/   one Parquet per model",
            "multi_label/    one Parquet per model",
            "schema.json     column types and original-label mapping",
            "manifest.json   source files, counts, hashes, and exclusions",
            "source_manifest.json  portable build inputs and per-model commands",
            "SHA256SUMS      release checksums",
            "```",
            "",
            "## Per-file provenance",
            "",
            "| File | Model | Strategy | Source lanes | Rows included | Rows excluded | Exact inference command | Exact projection command |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for item in manifest["files"]:
        lines.append(
            f"| `{item['relative_path']}` | `{item['model_id']}` | `{item['strategy']}` | {len(item['source_files'])} | {item['included_rows']} | {item['excluded_rows']} | `{item['inference_command']}` | `{item['projection_command']}` |"
        )
    lines.extend(
        [
            "",
            "## Validation policy",
            "",
            "Only rows with `validation_status=valid` and `final_status=completed` are exported. Single-label rows must have exactly one indicator set to 1 in their applicable label space. Multi-label rows may have zero or more indicators set to 1. Duplicate lane/row identities, malformed parsed JSON, unknown labels, and non-completed rows are excluded and counted in `manifest.json`.",
            "",
            "The MFT label space has ten values and the SHVT label space has twenty. Columns for the non-applicable space are zero. Cross-theory lanes remain separate through `lane_id`; repeated source text across lanes is therefore intentional and is not deduplicated across lanes.",
            "",
            "## Read a file",
            "",
            "```bash",
            "uv run python -c 'import pyarrow.parquet as pq; print(pq.read_table(\"single_label/<model>.parquet\").schema)'",
            "sha256sum single_label/*.parquet multi_label/*.parquet",
            "UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate_parsed_release.py release/protoethos-parsed-results-v1",
            "```",
            "",
            "## Release boundary",
            "",
            f"Generated at `{manifest['generated_at']}`. The release contains {len(manifest['files'])} Parquet files for {len(manifest['models'])} complete models. Partial model trees and the active Qwen3.8 recovery lane were excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def build(source_manifest_path: Path, output_root: Path, overwrite: bool) -> dict[str, Any]:
    """Build all model/strategy files and release metadata."""

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    models: list[str] = []
    for spec in source_manifest["models"]:
        model_id = str(spec["model_id"])
        model_slug = str(spec["model_slug"])
        if model_id not in models:
            models.append(model_id)
            source_root = Path(os.path.expandvars(str(spec.get("build_source_root", spec["source_root"]))))
        for strategy, directory in (("single_label_json", "single_label"), ("multi_label_json", "multi_label")):
            relative_path = Path(directory) / f"{model_slug}.parquet"
            output_file = output_root / relative_path
            summary = export_source(source_root, output_file, model_id, strategy, overwrite=overwrite)
            item = summary.as_dict()
            source_root_display = spec.get("source_root", str(source_root))
            item["source_root"] = source_root_display
            item["output_file"] = relative_path.as_posix()
            item["source_files"] = [
                Path(source_file).relative_to(summary.source_root).as_posix() for source_file in summary.source_files
            ]
            item["source_sha256"] = {
                Path(source_file).relative_to(summary.source_root).as_posix(): digest
                for source_file, digest in summary.source_sha256.items()
            }
            item.update(
                {
                    "relative_path": relative_path.as_posix(),
                    "source_origin": spec.get("source_origin", str(source_root)),
                    "source_acquisition_command": spec.get("source_acquisition_command"),
                    "runtime_configuration": _runtime_configuration(source_root),
                    "inference_command": spec["inference_command"],
                    "parse_steps": spec.get(
                        "parse_steps",
                        ["json_decode", "json_schema", "validation_status=valid", "final_status=completed"],
                    ),
                    "projection_command": _projection_command(
                        str(source_root_display), model_id, strategy, relative_path.as_posix()
                    ),
                }
            )
            files.append(item)
    generated_at = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    prompt_files = source_manifest.get("prompt_files", [])
    prompt_hashes = {path: _sha256(Path(path)) for path in prompt_files if Path(path).is_file()}
    manifest = {
        "release_version": "protoethos-parsed-results-v1",
        "generated_at": generated_at,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "build_command": "UV_CACHE_DIR=/tmp/uv-cache uv run python " + " ".join(sys.argv),
        "models": models,
        "strategies": ["single_label_json", "multi_label_json"],
        "prompt_files": prompt_files,
        "prompt_hashes": prompt_hashes,
        "files": files,
        "schema_file": "schema.json",
        "checksums_file": "SHA256SUMS",
        "selection_policy": "complete eleven-lane source roots only; valid completed parsed rows only",
    }
    (output_root / "schema.json").write_text(
        json.dumps(_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    build_command = manifest["build_command"]
    (output_root / "README.md").write_text(_readme(manifest, output_root, build_command), encoding="utf-8")
    checksum_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def refresh_metadata(source_manifest_path: Path, output_root: Path) -> dict[str, Any]:
    """Refresh release metadata without rereading or rewriting Parquet data."""

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    by_model = {str(spec["model_id"]): spec for spec in source_manifest["models"]}
    prompt_files = source_manifest.get("prompt_files", [])
    manifest["prompt_files"] = prompt_files
    manifest["prompt_hashes"] = {path: _sha256(Path(path)) for path in prompt_files if Path(path).is_file()}
    if not str(manifest.get("build_command", "")).startswith("UV_CACHE_DIR="):
        manifest["build_command"] = "UV_CACHE_DIR=/tmp/uv-cache uv run python " + str(manifest["build_command"])
    for item in manifest["files"]:
        spec = by_model[item["model_id"]]
        item["source_acquisition_command"] = spec.get("source_acquisition_command")
        build_source_root = Path(item["source_root"]).resolve()
        item["source_files"] = [
            Path(source_file).relative_to(build_source_root).as_posix()
            if Path(source_file).is_absolute()
            else Path(source_file).as_posix()
            for source_file in item["source_files"]
        ]
        item["source_sha256"] = {
            (
                Path(source_file).relative_to(build_source_root).as_posix()
                if Path(source_file).is_absolute()
                else Path(source_file).as_posix()
            ): digest
            for source_file, digest in item["source_sha256"].items()
        }
        item["source_root"] = spec["source_root"]
        item["output_file"] = item["relative_path"]
        build_source_root_path = Path(os.path.expandvars(str(spec.get("build_source_root", spec["source_root"]))))
        if build_source_root_path.is_dir():
            item["runtime_configuration"] = _runtime_configuration(build_source_root_path)
        item["projection_command"] = _projection_command(
            spec["source_root"], item["model_id"], item["strategy"], item["output_file"]
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "schema.json").write_text(
        json.dumps(_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "README.md").write_text(_readme(manifest, output_root, manifest["build_command"]), encoding="utf-8")
    checksum_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}")
    (output_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """Run the release builder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    manifest = (
        refresh_metadata(args.source_manifest, args.output_root)
        if args.metadata_only
        else build(args.source_manifest, args.output_root, args.overwrite)
    )
    print(json.dumps({"files": len(manifest["files"]), "models": len(manifest["models"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
