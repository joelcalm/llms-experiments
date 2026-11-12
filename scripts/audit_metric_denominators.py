#!/usr/bin/env python3
"""Check theory denominators and basic invariants in general metric exports.

The audit distinguishes theory-wide value counts from corpus coverage.  In
particular, MFRC is an MFT corpus with five represented value groups, so its
Recognition coverage is reported as 10/5 rather than 5/5.  Structure counts
are relation counts and are checked separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

THEORY_VALUE_COUNTS = {"mft": 10, "mfrc": 5, "shvt": 20}
DATASET_THEORIES = {"mftc": "mft", "mfrc": "mfrc", "valueeval": "shvt"}
STRUCTURE_RELATION_COUNTS = {"mft": 5, "mfrc": 0, "shvt": 30}
STRATEGIES = {"single_label_json", "multi_label_json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-axiom-csv", type=Path, required=True)
    parser.add_argument("--general-f1-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def in_unit_interval(value: str, *, allow_blank: bool = False) -> bool:
    if value == "" and allow_blank:
        return True
    number = float(value)
    return 0.0 <= number <= 1.0


def main() -> int:
    args = parse_args()
    axiom_rows = read_csv(args.general_axiom_csv)
    f1_rows = read_csv(args.general_f1_csv)
    errors: list[dict[str, Any]] = []

    axiom_keys = [(row["model"], row["dataset"], row["strategy"], row["axiom"]) for row in axiom_rows]
    f1_keys = [(row["model"], row["dataset"], row["strategy"]) for row in f1_rows]
    for name, keys in (("axiom", axiom_keys), ("classification", f1_keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        if duplicates:
            errors.append({"check": f"duplicate_{name}_keys", "keys": duplicates})

    expected_axioms = {"recognition", "structure"}
    coverage: Counter[tuple[str, int, int]] = Counter()
    for row in axiom_rows:
        dataset = row["dataset"]
        theory = DATASET_THEORIES.get(dataset)
        if theory is None:
            errors.append({"check": "unknown_dataset", "row": row})
            continue
        if row["strategy"] not in STRATEGIES or row["axiom"] not in expected_axioms:
            errors.append({"check": "unexpected_axiom_key", "row": row})
            continue
        if not in_unit_interval(row["score"], allow_blank=row["status"] == "not_evaluable"):
            errors.append({"check": "axiom_score_out_of_range", "row": row})
        count = int(row["evaluable_count"])
        if row["axiom"] == "recognition":
            total = THEORY_VALUE_COUNTS[theory]
            coverage[(dataset, total, count)] += 1
            if not 0 <= count <= total:
                errors.append(
                    {
                        "check": "recognition_coverage_out_of_range",
                        "dataset": dataset,
                        "theory": theory,
                        "theoretical_value_count": total,
                        "scored_value_count": count,
                    }
                )
        else:
            maximum = STRUCTURE_RELATION_COUNTS[theory]
            if not 0 <= count <= maximum:
                errors.append(
                    {
                        "check": "structure_relation_count_out_of_range",
                        "dataset": dataset,
                        "theory": theory,
                        "relation_count": count,
                        "maximum_relation_count": maximum,
                    }
                )

    for row in f1_rows:
        if row["dataset"] not in DATASET_THEORIES or row["strategy"] not in STRATEGIES:
            errors.append({"check": "unexpected_classification_key", "row": row})
            continue
        for metric in ("macro_f1", "micro_f1", "subset_accuracy", "macro_precision", "macro_recall"):
            if not in_unit_interval(row[metric]):
                errors.append({"check": "classification_score_out_of_range", "metric": metric, "row": row})
        if int(row["n_samples"]) <= 0:
            errors.append({"check": "non_positive_sample_count", "row": row})

    payload = {
        "status": "verified" if not errors else "failed",
        "policy": {
            "theory_value_counts": THEORY_VALUE_COUNTS,
            "dataset_theories": DATASET_THEORIES,
            "coverage_format": "theoretical total/scored values",
            "mfrc": "MFRC uses five foundation pairs and accepts either pole in each pair.",
            "structure": "evaluable_count is a relation count, not a value count.",
        },
        "rows": {"axioms": len(axiom_rows), "classification": len(f1_rows)},
        "recognition_coverage": [
            {
                "dataset": dataset,
                "theoretical_value_count": total,
                "scored_value_count": scored,
                "cells": cells,
            }
            for (dataset, total, scored), cells in sorted(coverage.items())
        ],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "errors": len(errors)}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
