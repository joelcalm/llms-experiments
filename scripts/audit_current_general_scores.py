#!/usr/bin/env python3
"""Independently audit current general-dataset metrics from raw Parquets.

The audit re-parses gold annotations and JSON predictions without importing
the evaluator's general-dataset runner. It checks the reported classification
metrics and worst-value Recognition score for MFTC, MFRC and ValueEval.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

MFT_LABELS = (
    "care",
    "harm",
    "fairness",
    "cheating",
    "loyalty",
    "betrayal",
    "authority",
    "subversion",
    "purity",
    "degradation",
)
SHVT_LABELS = (
    "self-direction: thought",
    "self-direction: action",
    "stimulation",
    "hedonism",
    "achievement",
    "power: dominance",
    "power: resources",
    "face",
    "security: personal",
    "security: societal",
    "tradition",
    "conformity: rules",
    "conformity: interpersonal",
    "humility",
    "benevolence: caring",
    "benevolence: dependability",
    "universalism: concern",
    "universalism: nature",
    "universalism: objectivity",
    "universalism: tolerance",
)
MFRC_LABELS = (
    "care_harm",
    "fairness_cheating",
    "loyalty_betrayal",
    "authority_subversion",
    "purity_degradation",
)
MFRC_LABEL_MAP = {
    "care": "care_harm",
    "harm": "care_harm",
    "fairness": "fairness_cheating",
    "cheating": "fairness_cheating",
    "loyalty": "loyalty_betrayal",
    "betrayal": "loyalty_betrayal",
    "authority": "authority_subversion",
    "subversion": "authority_subversion",
    "purity": "purity_degradation",
    "degradation": "purity_degradation",
}
INPUT_LABELS = {"mftc": MFT_LABELS, "mfrc": MFT_LABELS, "valueeval": SHVT_LABELS}
DATASET_LABELS = {"mftc": MFT_LABELS, "mfrc": MFRC_LABELS, "valueeval": SHVT_LABELS}
STRATEGIES = ("single_label_json", "multi_label_json")
MFRC_ALIASES = {"sanctity": "purity"}
THEORY_VALUE_COUNTS = {"mft": len(MFT_LABELS), "mfrc": len(MFRC_LABELS), "shvt": len(SHVT_LABELS)}
DATASET_THEORIES = {"mftc": "mft", "mfrc": "mfrc", "valueeval": "shvt"}
STRUCTURE_RELATION_COUNTS = {"mft": 5, "mfrc": 0, "shvt": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", required=True)
    parser.add_argument("--evaluator-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["model"], row["dataset"], row["strategy"]): row for row in csv.DictReader(handle)}


def load_axioms(path: Path) -> dict[tuple[str, str, str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        result: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
        for row in csv.DictReader(handle):
            key = (row["model"], row["dataset"], row["strategy"])
            result.setdefault(key, {})[row["axiom"]] = row
        return result


def labels_from_raw(raw: Any, dataset: str) -> frozenset[str]:
    if raw is None:
        return frozenset()
    values = json.loads(raw) if isinstance(raw, str) else raw
    aliases = MFRC_ALIASES if dataset == "mfrc" else {}
    taxonomy = frozenset(INPUT_LABELS[dataset])
    clean = {aliases.get(str(value).strip().lower(), str(value).strip().lower()) for value in values}
    clean &= taxonomy
    return frozenset(MFRC_LABEL_MAP[value] if dataset == "mfrc" else value for value in clean)


def prediction_from_row(row: dict[str, Any], strategy: str, dataset: str) -> frozenset[str] | None:
    try:
        payload = json.loads(str(row["parsed_output"]))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    taxonomy = frozenset(INPUT_LABELS[dataset])
    if strategy == "single_label_json":
        label = payload.get("label")
        if not isinstance(label, str) or label.strip().lower() not in taxonomy:
            return None
        clean = label.strip().lower()
        return frozenset({MFRC_LABEL_MAP[clean] if dataset == "mfrc" else clean})
    values = payload.get("labels")
    if not isinstance(values, list):
        return None
    labels = [str(value).strip().lower() for value in values]
    if len(set(labels)) != len(labels) or not set(labels) <= taxonomy:
        return None
    return frozenset(MFRC_LABEL_MAP[label] if dataset == "mfrc" else label for label in labels)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def audit_cell(
    roots: list[Path],
    classification: dict[tuple[str, str, str], dict[str, str]],
    axioms: dict[tuple[str, str, str], dict[tuple[str, str], dict[str, str]]],
    model: str,
    dataset: str,
    strategy: str,
) -> dict[str, Any]:
    path = next(
        (
            root / model / f"dataset={dataset}" / f"{strategy}.parquet"
            for root in roots
            if (root / model / f"dataset={dataset}" / f"{strategy}.parquet").is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(f"no raw lane for {model} {dataset} {strategy}")
    table = pq.read_table(
        path,
        columns=["input_row_id", "gold_labels", "parsed_output", "final_status", "validation_status"],
    )
    rows: list[tuple[frozenset[str], frozenset[str]]] = []
    seen: set[str] = set()
    incomplete = duplicate = malformed = 0
    for raw in table.to_pylist():
        if raw["final_status"] != "completed" or raw["validation_status"] != "valid":
            incomplete += 1
            continue
        identity = str(raw["input_row_id"]).strip()
        if identity in seen:
            duplicate += 1
            continue
        prediction = prediction_from_row(raw, strategy, dataset)
        if prediction is None:
            malformed += 1
            continue
        seen.add(identity)
        rows.append((labels_from_raw(raw["gold_labels"], dataset), prediction))

    taxonomy = DATASET_LABELS[dataset]
    counts = {label: [0, 0, 0] for label in taxonomy}
    exact = 0
    for gold, predicted in rows:
        exact += gold == predicted
        for label in gold & predicted:
            counts[label][0] += 1
        for label in predicted - gold:
            counts[label][1] += 1
        for label in gold - predicted:
            counts[label][2] += 1
    per_label = [prf(*counts[label]) for label in taxonomy]
    micro = prf(
        sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values()), sum(c[2] for c in counts.values())
    )
    expected = {
        "macro_f1": sum(value[2] for value in per_label) / len(per_label),
        "micro_f1": micro[2],
        "subset_accuracy": exact / len(rows) if rows else 0.0,
        "macro_precision": sum(value[0] for value in per_label) / len(per_label),
        "macro_recall": sum(value[1] for value in per_label) / len(per_label),
        "n_samples": len(rows),
    }
    recognition = []
    for label in taxonomy:
        members = [predicted for gold, predicted in rows if label in gold]
        recognition.append(sum(label in predicted for predicted in members) / len(members) if members else None)
    expected["recognition"] = (
        min(value for value in recognition if value is not None)
        if any(value is not None for value in recognition)
        else None
    )

    reported_classification = classification[(model, dataset, strategy)]
    reported_axioms = axioms[(model, dataset, strategy)]
    mismatches: list[dict[str, Any]] = []
    theory = DATASET_THEORIES[dataset]
    theoretical_value_count = THEORY_VALUE_COUNTS[theory]
    recognition_scored = int(reported_axioms["recognition"]["evaluable_count"])
    structure_relations = int(reported_axioms["structure"]["evaluable_count"])
    denominator_checks = {
        "theory": theory,
        "theoretical_value_count": theoretical_value_count,
        "recognition_scored_value_count": recognition_scored,
        "structure_relation_count": structure_relations,
        "recognition_coverage_valid": 0 <= recognition_scored <= theoretical_value_count,
        "structure_relation_count_valid": 0 <= structure_relations <= STRUCTURE_RELATION_COUNTS[theory],
        "classification_scores_bounded": all(
            0.0 <= float(reported_classification[key]) <= 1.0
            for key in ("macro_f1", "micro_f1", "subset_accuracy", "macro_precision", "macro_recall")
        ),
        "axiom_scores_bounded": all(
            row["score"] == "" or 0.0 <= float(row["score"]) <= 1.0 for row in reported_axioms.values()
        ),
    }
    for check, passed in denominator_checks.items():
        if check in {"theory", "theoretical_value_count", "recognition_scored_value_count", "structure_relation_count"}:
            continue
        if not passed:
            mismatches.append({"metric": check, "expected": True, "reported": passed})
    for key, value in expected.items():
        if key == "recognition":
            reported = float(reported_axioms["recognition"]["score"])
            matches = value is not None and abs(value - reported) < 1e-12
        elif key == "n_samples":
            reported = int(reported_classification[key])
            matches = value == reported
        else:
            reported = float(reported_classification[key])
            matches = abs(value - reported) < 1e-12
        if not matches:
            mismatches.append({"metric": key, "expected": value, "reported": reported})
    return {
        "model": model,
        "dataset": dataset,
        "strategy": strategy,
        "source_rows": table.num_rows,
        "used_rows": len(rows),
        "incomplete_rows": incomplete,
        "duplicate_ids": duplicate,
        "malformed_predictions": malformed,
        "denominator_checks": denominator_checks,
        "mismatches": mismatches,
    }


def main() -> int:
    args = parse_args()
    classification = load_csv(args.evaluator_output / "general_classification_f1.csv")
    axioms = load_axioms(args.evaluator_output / "general_axiom_scores.csv")
    models = sorted({key[0] for key in classification})
    cells = [
        audit_cell(args.results_root, classification, axioms, model, dataset, strategy)
        for model in models
        for dataset in DATASET_LABELS
        for strategy in STRATEGIES
    ]
    mismatches = [item for cell in cells for item in cell["mismatches"]]
    payload = {
        "status": "verified" if not mismatches else "mismatch_found",
        "cells": cells,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cells": len(cells), "mismatches": len(mismatches)}))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
