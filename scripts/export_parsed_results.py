#!/usr/bin/env python3
"""Export validated JSON predictions to compact, shareable Parquet files.

The source files remain Result Contract 2.0 projections.  This exporter keeps
only stable row identity and the parsed label set, represented as one uint8
indicator column per value.  It deliberately excludes raw text, raw model
responses, timing data, confidence data, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MFT_LABELS = (
    "authority",
    "betrayal",
    "care",
    "cheating",
    "degradation",
    "fairness",
    "harm",
    "loyalty",
    "purity",
    "subversion",
)
SHVT_LABELS = (
    "Achievement",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Conformity: interpersonal",
    "Conformity: rules",
    "Face",
    "Hedonism",
    "Humility",
    "Power: dominance",
    "Power: resources",
    "Security: personal",
    "Security: societal",
    "Self-direction: action",
    "Self-direction: thought",
    "Stimulation",
    "Tradition",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: objectivity",
    "Universalism: tolerance",
)

LANE_LABEL_SPACE = {
    "mftc": "mft10",
    "mfrc": "mft10",
    "protoethos_mft": "mft10",
    "protoethos_mft_transformed": "mft10",
    "protoethos_shvt_core_cross_mft": "mft10",
    "valueeval": "shvt20",
    "protoethos_shvt": "shvt20",
    "protoethos_shvt_transformed": "shvt20",
    "protoethos_diagnostic_mft": "mft10",
    "protoethos_diagnostic_shvt": "shvt20",
    "protoethos_mft_core_cross_shvt": "shvt20",
}


def column_name(label_space: str, label: str) -> str:
    """Return a portable, stable Parquet column name for an original label."""

    prefix = "mft" if label_space == "mft10" else "shvt"
    name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{prefix}__{name}"


VALUE_COLUMNS = tuple(column_name("mft10", label) for label in MFT_LABELS) + tuple(
    column_name("shvt20", label) for label in SHVT_LABELS
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("model_id", pa.string()),
        ("strategy", pa.string()),
        ("lane_id", pa.string()),
        ("label_space", pa.string()),
        ("input_row_id", pa.string()),
        ("source_position", pa.int64()),
        *((name, pa.uint8()) for name in VALUE_COLUMNS),
    ]
)


@dataclass(frozen=True)
class ExportSummary:
    """Quality and provenance summary for one compact Parquet projection."""

    source_root: str
    output_file: str
    model_id: str
    strategy: str
    source_files: tuple[str, ...]
    source_rows: int
    included_rows: int
    excluded_rows: int
    exclusion_reasons: dict[str, int]
    lane_rows: dict[str, int]
    source_sha256: dict[str, str]
    output_sha256: str
    output_rows: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""

        return {
            "source_root": self.source_root,
            "output_file": self.output_file,
            "model_id": self.model_id,
            "strategy": self.strategy,
            "source_files": list(self.source_files),
            "source_rows": self.source_rows,
            "included_rows": self.included_rows,
            "excluded_rows": self.excluded_rows,
            "exclusion_reasons": self.exclusion_reasons,
            "lane_rows": self.lane_rows,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "output_rows": self.output_rows,
        }


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels_for_lane(lane_id: str) -> tuple[str, tuple[str, ...]]:
    try:
        label_space = LANE_LABEL_SPACE[lane_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported lane without a label-space mapping: {lane_id}") from exc
    return label_space, MFT_LABELS if label_space == "mft10" else SHVT_LABELS


def _parsed_labels(row: dict[str, Any], strategy: str, allowed: set[str]) -> list[str]:
    parsed = json.loads(row.get("parsed_output") or "null")
    if strategy == "single_label_json":
        if not isinstance(parsed, dict) or not isinstance(parsed.get("label"), str):
            raise ValueError("single-label parsed output does not contain a string label")
        labels = [parsed["label"]]
    elif strategy == "multi_label_json":
        if not isinstance(parsed, dict) or not isinstance(parsed.get("labels"), list):
            raise ValueError("multi-label parsed output does not contain a labels list")
        labels = parsed["labels"]
        if not all(isinstance(label, str) for label in labels):
            raise ValueError("multi-label parsed output contains a non-string label")
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if len(set(labels)) != len(labels):
        raise ValueError("parsed output contains duplicate labels")
    if not set(labels).issubset(allowed):
        raise ValueError("parsed output contains a label outside the lane label space")
    if strategy == "single_label_json" and len(labels) != 1:
        raise ValueError("single-label parsed output must contain exactly one label")
    return labels


def _compact_row(
    row: dict[str, Any],
    model_id: str,
    strategy: str,
    lane_id: str,
    label_space: str,
    labels: list[str],
) -> dict[str, Any]:
    """Convert one validated Result Contract row to the compact schema."""

    label_columns = {column_name(label_space, label) for label in labels}
    output = {
        "model_id": model_id,
        "strategy": strategy,
        "lane_id": lane_id,
        "label_space": label_space,
        "input_row_id": str(row["input_row_id"]),
        "source_position": int(row["source_position"]),
    }
    output.update({name: int(name in label_columns) for name in VALUE_COLUMNS})
    return output


def export_source(
    source_root: Path,
    output_file: Path,
    model_id: str,
    strategy: str,
    *,
    require_eleven_lanes: bool = True,
    overwrite: bool = False,
) -> ExportSummary:
    """Export one model/strategy from final lane Parquets.

    Only rows with ``validation_status=valid`` and ``final_status=completed``
    are emitted.  Invalid, incomplete, malformed, duplicate, and out-of-space
    labels are counted and excluded.  A complete release source must contain
    all eleven lanes unless ``require_eleven_lanes`` is disabled.
    """

    source_root = source_root.resolve()
    output_file = output_file.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite explicitly: {output_file}")
    lane_dirs = sorted(d for d in source_root.glob("dataset=*") if d.is_dir())
    expected_lanes = set(LANE_LABEL_SPACE)
    actual_lanes = {d.name.removeprefix("dataset=") for d in lane_dirs}
    if require_eleven_lanes and actual_lanes != expected_lanes:
        missing = sorted(expected_lanes - actual_lanes)
        extra = sorted(actual_lanes - expected_lanes)
        raise ValueError(f"Source root is not a complete eleven-lane tree; missing={missing}, extra={extra}")
    if not lane_dirs:
        raise ValueError(f"No dataset=* lane directories found under {source_root}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    source_files: list[str] = []
    source_sha256: dict[str, str] = {}
    source_rows = 0
    included_rows = 0
    exclusion_reasons: Counter[str] = Counter()
    lane_rows: Counter[str] = Counter()
    try:
        for lane_dir in lane_dirs:
            lane_id = lane_dir.name.removeprefix("dataset=")
            if lane_id not in LANE_LABEL_SPACE:
                continue
            source_file = lane_dir / f"{strategy}.parquet"
            manifest = lane_dir / "manifest.json"
            if not source_file.is_file() or source_file.stat().st_size == 0:
                raise FileNotFoundError(f"Missing final {strategy} Parquet: {source_file}")
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing lane manifest: {manifest}")
            source_files.append(str(source_file))
            source_sha256[str(source_file)] = sha256_file(source_file)
            label_space, allowed_labels = _labels_for_lane(lane_id)
            seen_keys: set[tuple[str, str, int]] = set()
            parquet = pq.ParquetFile(source_file)
            for batch in parquet.iter_batches(
                columns=["input_row_id", "source_position", "parsed_output", "validation_status", "final_status"],
                batch_size=65536,
            ):
                rows = batch.to_pylist()
                source_rows += len(rows)
                compact_rows: list[dict[str, Any]] = []
                for row in rows:
                    if row.get("validation_status") != "valid":
                        exclusion_reasons["validation_status_not_valid"] += 1
                        continue
                    if row.get("final_status") not in {"completed", "complete"}:
                        exclusion_reasons["final_status_not_completed"] += 1
                        continue
                    try:
                        source_position = int(row["source_position"])
                        input_row_id = str(row["input_row_id"])
                    except (KeyError, TypeError, ValueError):
                        exclusion_reasons["missing_row_identity"] += 1
                        continue
                    key = (lane_id, input_row_id, source_position)
                    if key in seen_keys:
                        exclusion_reasons["duplicate_row_identity"] += 1
                        continue
                    seen_keys.add(key)
                    try:
                        labels = _parsed_labels(row, strategy, set(allowed_labels))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        exclusion_reasons["parsed_output_invalid"] += 1
                        continue
                    compact_rows.append(_compact_row(row, model_id, strategy, lane_id, label_space, labels))
                if compact_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary,
                            OUTPUT_SCHEMA,
                            compression="zstd",
                            compression_level=3,
                            use_dictionary=["model_id", "strategy", "lane_id", "label_space", "input_row_id"],
                            write_statistics=True,
                        )
                    writer.write_table(pa.Table.from_pylist(compact_rows, schema=OUTPUT_SCHEMA))
                    included_rows += len(compact_rows)
                    lane_rows[lane_id] += len(compact_rows)
        if writer is None:
            raise ValueError("No valid parsed rows were available for export")
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_file)
    output_sha256 = sha256_file(output_file)
    output_rows = pq.ParquetFile(output_file).metadata.num_rows
    if output_rows != included_rows:
        raise AssertionError(f"Output row count mismatch: metadata={output_rows}, counted={included_rows}")
    return ExportSummary(
        source_root=str(source_root),
        output_file=str(output_file),
        model_id=model_id,
        strategy=strategy,
        source_files=tuple(source_files),
        source_rows=source_rows,
        included_rows=included_rows,
        excluded_rows=source_rows - included_rows,
        exclusion_reasons=dict(sorted(exclusion_reasons.items())),
        lane_rows=dict(sorted(lane_rows.items())),
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        output_rows=output_rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--strategy", choices=("single_label_json", "multi_label_json"), required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Allow fewer than the complete eleven lanes")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    """Run the command-line exporter."""

    args = _parser().parse_args()
    summary = export_source(
        args.source_root,
        args.output_file,
        args.model_id,
        args.strategy,
        require_eleven_lanes=not args.allow_partial,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
