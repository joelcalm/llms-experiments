"""Compute paired prediction agreement between two result roots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pyarrow import parquet


@dataclass(frozen=True)
class AgreementMetrics:
    """Agreement and coverage statistics for one model, dataset, and strategy."""

    model: str
    dataset: str
    strategy: str
    bf16_rows: int
    int4_rows: int
    matched_rows: int
    duplicate_bf16_keys: int
    duplicate_int4_keys: int
    bf16_invalid: int
    int4_invalid: int
    both_valid: int
    exact_agreement_valid_pct: float | None
    exact_agreement_all_matched_pct: float | None
    mean_jaccard_valid_pct: float | None
    labelwise_agreement_valid_pct: float | None


def _key(row: dict[str, Any]) -> tuple[str, int]:
    """Return the stable row key used to pair repeated runs."""

    return (str(row["input_row_id"]), int(row.get("source_position") or 0))


def _prediction(row: dict[str, Any], strategy: str) -> frozenset[str] | None:
    """Canonicalise a validated JSON prediction to a label set."""

    value = row.get("parsed_output")
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    if strategy == "single_label_json":
        label = parsed.get("label")
        return frozenset({label}) if isinstance(label, str) and label else None
    labels = parsed.get("labels")
    if not isinstance(labels, list) or not all(isinstance(label, str) and label for label in labels):
        return None
    return frozenset(labels)


def _valid(row: dict[str, Any], strategy: str) -> bool:
    """Return whether a row has a completed, schema-valid prediction."""

    return (
        row.get("validation_status") == "valid"
        and row.get("final_status") == "completed"
        and _prediction(row, strategy) is not None
    )


def _unique_rows(rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    """Index only unique composite keys and return the number of duplicate keys."""

    counts = Counter(_key(row) for row in rows)
    return ({_key(row): row for row in rows if counts[_key(row)] == 1}, sum(count > 1 for count in counts.values()))


def compare(model: str, bf16_path: Path, int4_path: Path, dataset: str, strategy: str) -> AgreementMetrics:
    """Compare one pair of Parquet files at the input-row grain."""

    bf16_rows = parquet.read_table(bf16_path).to_pylist()
    int4_rows = parquet.read_table(int4_path).to_pylist()
    bf16, bf16_duplicates = _unique_rows(bf16_rows)
    int4, int4_duplicates = _unique_rows(int4_rows)
    keys = sorted(set(bf16) & set(int4))
    bf16_valid = {key for key in keys if _valid(bf16[key], strategy)}
    int4_valid = {key for key in keys if _valid(int4[key], strategy)}
    both_valid = sorted(bf16_valid & int4_valid)
    exact = sum(_prediction(bf16[key], strategy) == _prediction(int4[key], strategy) for key in both_valid)
    jaccards: list[float] = []
    labels = {label for key in keys for label in (bf16[key].get("gold_labels") or [])}
    label_agreements = 0
    for key in both_valid:
        bf16_prediction = _prediction(bf16[key], strategy)
        int4_prediction = _prediction(int4[key], strategy)
        assert bf16_prediction is not None and int4_prediction is not None
        union = bf16_prediction | int4_prediction
        jaccards.append(len(bf16_prediction & int4_prediction) / len(union) if union else 1.0)
        label_agreements += sum((label in bf16_prediction) == (label in int4_prediction) for label in labels)
    total_label_decisions = len(both_valid) * len(labels)
    return AgreementMetrics(
        model=model,
        dataset=dataset,
        strategy=strategy,
        bf16_rows=len(bf16_rows),
        int4_rows=len(int4_rows),
        matched_rows=len(keys),
        duplicate_bf16_keys=bf16_duplicates,
        duplicate_int4_keys=int4_duplicates,
        bf16_invalid=len(keys) - len(bf16_valid),
        int4_invalid=len(keys) - len(int4_valid),
        both_valid=len(both_valid),
        exact_agreement_valid_pct=100 * exact / len(both_valid) if both_valid else None,
        exact_agreement_all_matched_pct=100 * exact / len(keys) if keys else None,
        mean_jaccard_valid_pct=100 * sum(jaccards) / len(jaccards) if jaccards else None,
        labelwise_agreement_valid_pct=100 * label_agreements / total_label_decisions if total_label_decisions else None,
    )


def label_agreement_rows(
    model: str,
    bf16_path: Path,
    int4_path: Path,
    dataset: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Return one binary-agreement row for every observed label.

    The denominator is the set of rows whose predictions are valid in both
    runs. Labels are taken from the gold labels and valid predictions so a
    label produced only by one run is still represented.
    """

    bf16_rows = parquet.read_table(bf16_path).to_pylist()
    int4_rows = parquet.read_table(int4_path).to_pylist()
    bf16, _ = _unique_rows(bf16_rows)
    int4, _ = _unique_rows(int4_rows)
    keys = sorted(set(bf16) & set(int4))
    both_valid = [key for key in keys if _valid(bf16[key], strategy) and _valid(int4[key], strategy)]
    labels = {str(label) for key in keys for label in (bf16[key].get("gold_labels") or []) if isinstance(label, str)}
    for key in both_valid:
        bf16_prediction = _prediction(bf16[key], strategy)
        int4_prediction = _prediction(int4[key], strategy)
        assert bf16_prediction is not None and int4_prediction is not None
        labels.update(bf16_prediction | int4_prediction)
    prediction_pairs: list[tuple[frozenset[str], frozenset[str]]] = []
    for key in both_valid:
        bf16_prediction = _prediction(bf16[key], strategy)
        int4_prediction = _prediction(int4[key], strategy)
        if bf16_prediction is None or int4_prediction is None:
            raise AssertionError("both_valid contained an invalid prediction")
        prediction_pairs.append((bf16_prediction, int4_prediction))

    rows: list[dict[str, Any]] = []
    for label in sorted(labels):
        agreement = sum(
            (label in bf16_prediction) == (label in int4_prediction)
            for bf16_prediction, int4_prediction in prediction_pairs
        )
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "strategy": strategy,
                "label": label,
                "both_valid": len(both_valid),
                "agreement_pct": 100 * agreement / len(both_valid) if both_valid else None,
            }
        )
    return rows


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("MODEL", "BF16_ROOT", "INT4_ROOT"),
        required=True,
        help="Model display name followed by its BF16 and INT4 model roots.",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        help="Optional CSV path for label-wise binary agreement rows.",
    )
    return parser.parse_args()


def main() -> None:
    """Read paired model roots and write a CSV agreement report."""

    args = _arguments()
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for model, bf16_root_text, int4_root_text in args.pair:
        bf16_root = Path(bf16_root_text)
        int4_root = Path(int4_root_text)
        for dataset in ("mftc", "mfrc", "valueeval"):
            for strategy in ("single_label_json", "multi_label_json"):
                bf16_path = bf16_root / f"dataset={dataset}" / f"{strategy}.parquet"
                int4_path = int4_root / f"dataset={dataset}" / f"{strategy}.parquet"
                metrics = compare(
                    model,
                    bf16_path,
                    int4_path,
                    dataset,
                    strategy,
                )
                rows.append(asdict(metrics))
                label_rows.extend(label_agreement_rows(model, bf16_path, int4_path, dataset, strategy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.label_output is not None:
        args.label_output.parent.mkdir(parents=True, exist_ok=True)
        with args.label_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
            writer.writeheader()
            writer.writerows(label_rows)


if __name__ == "__main__":
    main()
