#!/usr/bin/env python3
"""Build a post-shutdown AIS execution matrix from canonical and recovered data.

The canonical snapshot is authoritative for the ordinary result trees.  The
recovery overlay adds result trees written by replacement jobs which were not
part of the canonical collector's two historical roots.  A lane is complete
only when its manifest and both JSON Parquet variants are present and
non-empty.  The generated CSV/JSON files are report inputs, not metric data.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

GENERAL = ("mftc", "mfrc", "valueeval")
PROTOETHOS = (
    "protoethos_mft",
    "protoethos_shvt",
    "protoethos_mft_transformed",
    "protoethos_shvt_transformed",
    "protoethos_diagnostic_mft",
    "protoethos_diagnostic_shvt",
    "protoethos_mft_core_cross_shvt",
    "protoethos_shvt_core_cross_mft",
)
LANES = GENERAL + PROTOETHOS
INPUT_ROWS = {
    "mftc": 34_987,
    "mfrc": 17_886,
    "valueeval": 8_865,
    "protoethos_mft": 200_000,
    "protoethos_shvt": 400_000,
    "protoethos_mft_transformed": 650_000,
    "protoethos_shvt_transformed": 1_300_000,
    "protoethos_diagnostic_mft": 450_000,
    "protoethos_diagnostic_shvt": 450_000,
    "protoethos_mft_core_cross_shvt": 200_000,
    "protoethos_shvt_core_cross_mft": 400_000,
}

# These are replacement trees outside the two canonical roots scanned by the
# historical AIS collector.  Multiple paths are intentional: the union keeps
# the best valid lane while retaining the incomplete trees for provenance.
OVERLAYS = {
    "openai_gpt_oss_20b": (
        "results_json_only_gpt_oss_20b_token_recovery_full_20260816/openai_gpt_oss_20b_token_recovery_full_20260816",
    ),
    "qwen_qwen3_8_27b_fp8": ("results_json_only_fp8_qwen38_full_20260818/qwen_qwen3_8_27b_fp8_full_20260818",),
    "qwen_qwen3_6_27b": ("results_json_only_fp8_semantic_repairs_20260816/qwen_qwen3_6_27b_fp8_shvt_repair",),
    "qwen_qwen3_6_35b_a3b": (
        "results_json_only_fp8_semantic_repairs_20260816/qwen_qwen3_6_35b_a3b_fp8_semantic_repair",
    ),
    "meta_models_muse_glimmer_30b": (
        "results_json_only_fp8_muse_native_container_full_20260814/"
        "redhatai_muse_glimmer_30b_fp8_block_native_container_full_20260814",
        "results_json_only_fp8_muse_native_container_full_20260814_"
        "repaired_20260817/"
        "redhatai_muse_glimmer_30b_fp8_block_native_container_full_20260814_"
        "repaired_20260817",
        "results_json_only_fp8_semantic_repairs_20260816/redhatai_muse_glimmer_30b_fp8_block_shvt_repair",
    ),
}


def complete_lane(directory: Path) -> bool:
    return all(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in ("manifest.json", "single_label_json.parquet", "multi_label_json.parquet")
    )


def event_progress(directory: Path) -> tuple[int, float]:
    rows = 0
    timestamps: list[float] = []
    for path in directory.glob("logs/*.events.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "batch_completed":
                rows += int(event.get("rows", 0) or 0)
                if event.get("timestamp") is not None:
                    timestamps.append(float(event["timestamp"]))
    active_seconds = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    return rows, active_seconds


def overlay_record(directory: Path, model: str, lane: str) -> dict[str, Any] | None:
    directory = directory / f"dataset={lane}"
    if not directory.exists():
        return None
    rows, seconds = event_progress(directory)
    done = complete_lane(directory)
    has_parts = any(directory.glob("**/part-*.parquet"))
    return {
        "active_seconds": round(seconds, 3),
        "complete": done,
        "input_rows": INPUT_ROWS[lane],
        "lane": lane,
        "model": model,
        "rows_done": INPUT_ROWS[lane] * 2 if done else min(rows, INPUT_ROWS[lane] * 2),
        "started": done or bool(rows or has_parts),
        "target_rows": INPUT_ROWS[lane] * 2,
        "tree": "recovered_overlay",
    }


def merge(canonical: dict[str, Any], recovery_root: Path) -> list[dict[str, Any]]:
    records = {(item["model"], item["lane"]): dict(item) for item in canonical["lanes"]}
    for model, relative_roots in OVERLAYS.items():
        for lane in LANES:
            candidates = [
                overlay_record(recovery_root / relative_root, model, lane) for relative_root in relative_roots
            ]
            candidates = [candidate for candidate in candidates if candidate is not None]
            if not candidates:
                continue
            best = max(candidates, key=lambda item: (item["complete"], item["rows_done"]))
            current = records.get((model, lane))
            if current is None or best["complete"] or best["rows_done"] > current["rows_done"]:
                records[(model, lane)] = best
    return sorted(records.values(), key=lambda item: (item["model"], LANES.index(item["lane"])))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("&", r"\&")


def write_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabularx}{\linewidth}{@{}L{0.43\linewidth} r r r@{}}",
        r"\toprule",
        r"Model & Complete & Partial & Missing\\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['model'])} & {row['complete_lanes']} & "
            f"{row['partial_lanes']} & {row['missing_lanes']}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    lanes = merge(canonical, args.recovery_root)
    models = sorted({item["model"] for item in lanes})
    by_key = {(item["model"], item["lane"]): item for item in lanes}

    matrix: list[dict[str, Any]] = []
    eta: list[dict[str, Any]] = []
    rates: dict[str, float] = {}
    model_records: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        model_records[model] = [by_key[(model, lane)] for lane in LANES]
        active_rows = sum(item["rows_done"] for item in model_records[model] if item["active_seconds"])
        active_seconds = sum(item["active_seconds"] for item in model_records[model] if item["active_seconds"])
        if active_seconds:
            rates[model] = active_rows / active_seconds
    observed = sorted(rates.values())
    if observed:
        midpoint = len(observed) // 2
        low_rates = observed[:midpoint] or observed
        high_rates = observed[midpoint:] or observed
        fallback_low = median(low_rates)
        fallback_high = median(high_rates)
    else:
        fallback_low = fallback_high = 1.0
    for model in models:
        model_lanes = model_records[model]
        complete = [item for item in model_lanes if item["complete"]]
        partial = [item for item in model_lanes if item["started"] and not item["complete"]]
        missing = [item for item in model_lanes if not item["started"]]
        matrix.append(
            {
                "model": model,
                "complete_lanes": len(complete),
                "partial_lanes": len(partial),
                "missing_lanes": len(missing),
                "completed": ";".join(item["lane"] for item in complete),
                "partial": ";".join(item["lane"] for item in partial),
                "missing": ";".join(item["lane"] for item in missing),
                "complete_matrix": len(complete) == len(LANES),
            }
        )
        rate = rates.get(model, 0.0)
        remaining = sum(item["target_rows"] - item["rows_done"] for item in model_lanes)
        low_hours = remaining / (rate or fallback_high) / 3600
        high_hours = remaining / (rate or fallback_low) / 3600
        eta.append(
            {
                "model": model,
                "rows_remaining": remaining,
                "lanes_remaining": len(partial) + len(missing),
                "measured_rows_per_second": round(rate, 3) if rate else "",
                "gpu_hours_at_measured_rate": round(remaining / rate / 3600, 2) if rate else "",
                "gpu_hours_low": round(low_hours, 2),
                "gpu_hours_high": round(high_hours, 2),
                "estimate_basis": "measured" if rate else "observed-rate range",
                "gpu_class": (
                    "H100 (queued)"
                    if "llama_4_scout" in model
                    else "A100 (queued)"
                    if model
                    in {
                        "qwen_qwen3_6_35b_a3b",
                        "meta_models_muse_glimmer_30b",
                        "qwen_qwen3_8_27b_fp8",
                    }
                    else "not queued"
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "generated_at": datetime.now(UTC).isoformat(),
        "canonical_snapshot": str(args.canonical),
        "recovery_root": str(args.recovery_root),
        "expected_lanes": list(LANES),
        "variants": ["single_label_json", "multi_label_json"],
        "models": models,
        "lanes": lanes,
        "matrix": matrix,
        "eta": eta,
        "notes": [
            "A lane is complete only with manifest.json and both final JSON Parquets.",
            "Recovery overlays are unioned with canonical trees and retain partial progress.",
            "GPU-hour estimates are compute-only and exclude Condor queue wait after the heatwave.",
        ],
    }
    (args.out_dir / "execution_post_shutdown_20260820.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.out_dir / "execution_matrix_20260820.csv", list(matrix[0]), matrix)
    write_csv(args.out_dir / "execution_eta_20260820.csv", list(eta[0]), eta)
    write_csv(args.out_dir / "execution_lanes_20260820.csv", list(lanes[0]), lanes)
    write_summary_table(args.out_dir.parent / "tables" / "execution_post_shutdown_summary.tex", matrix)
    print(json.dumps({"models": len(models), "lanes": len(lanes), "complete": sum(item["complete"] for item in lanes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
