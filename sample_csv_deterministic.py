#!/usr/bin/env python3
"""Create a deterministic random sample of CSV rows by text column.

Sampling is reproducible for the same input ordering + seed.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


def detect_csv_dialect(csv_path: Path) -> csv.Dialect:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def count_eligible_rows(input_csv: Path, text_column: str, dialect: csv.Dialect) -> tuple[int, list[str]]:
    with input_csv.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        if text_column not in reader.fieldnames:
            raise ValueError(f"Missing text column '{text_column}' in input CSV")

        count = 0
        for row in reader:
            if (row.get(text_column) or "").strip():
                count += 1
        return count, list(reader.fieldnames)


def write_sample(
    input_csv: Path,
    output_csv: Path,
    text_column: str,
    selected_positions: list[int],
    dialect: csv.Dialect,
    fieldnames: list[str],
) -> int:
    selected_iter = iter(selected_positions)
    next_selected = next(selected_iter, None)
    written = 0
    eligible_pos = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as in_f, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f, dialect=dialect)
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            text = (row.get(text_column) or "").strip()
            if not text:
                continue

            if next_selected is None:
                break

            if eligible_pos == next_selected:
                writer.writerow(row)
                written += 1
                next_selected = next(selected_iter, None)

            eligible_pos += 1

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic random CSV sampler")
    ap.add_argument("--input-csv", required=True, help="Input CSV path")
    ap.add_argument("--output-csv", required=True, help="Output sampled CSV path")
    ap.add_argument("--text-column", default="text", help="CSV text column used to filter empty rows")
    ap.add_argument("--sample-size", type=int, required=True, help="Target sample size")
    ap.add_argument("--seed", type=int, default=20260417, help="Deterministic sampling seed")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    args = ap.parse_args()

    if args.sample_size <= 0:
        print("ERROR: --sample-size must be > 0", file=sys.stderr)
        return 1

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.exists():
        print(f"ERROR: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    if output_csv.exists() and not args.overwrite:
        print(f"Output already exists, not overwriting: {output_csv}")
        return 0

    dialect = detect_csv_dialect(input_csv)
    total_eligible, fieldnames = count_eligible_rows(input_csv, args.text_column, dialect)

    if total_eligible == 0:
        print("ERROR: no non-empty rows found in input", file=sys.stderr)
        return 1

    sample_n = min(args.sample_size, total_eligible)
    rng = random.Random(args.seed)
    selected_positions = sorted(rng.sample(range(total_eligible), sample_n))

    written = write_sample(
        input_csv=input_csv,
        output_csv=output_csv,
        text_column=args.text_column,
        selected_positions=selected_positions,
        dialect=dialect,
        fieldnames=fieldnames,
    )

    print(f"Input eligible rows: {total_eligible}")
    print(f"Seed: {args.seed}")
    print(f"Requested sample size: {args.sample_size}")
    print(f"Written rows: {written}")
    print(f"Output: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
